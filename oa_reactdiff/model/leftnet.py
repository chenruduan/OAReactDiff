import math
from math import pi
from typing import Optional, Tuple, Callable

import torch
import numpy as np
from torch import nn, Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter, scatter_mean

from oa_reactdiff.model.util_funcs import unsorted_segment_sum
from oa_reactdiff.model.core import MLP

EPS = 1e-6


def swish(x):
    return x * torch.sigmoid(x)


def com(x):
    return x - torch.mean(x, dim=0)


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x


class RBFEmb(nn.Module):
    r"""
    radial basis function to embed distances
    modified: delete cutoff with r
    """

    def __init__(self, num_rbf, rbound_upper, rbf_trainable=False):
        super().__init__()
        self.rbound_upper = rbound_upper
        self.rbound_lower = 0
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()

        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.rbound_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.rbound_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (end_value - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        rbounds = 0.5 * (torch.cos(dist * pi / self.rbound_upper) + 1.0)
        rbounds = rbounds * (dist < self.rbound_upper).float()
        return rbounds * torch.exp(
            -self.betas * torch.square((torch.exp(-dist) - self.means))
        )


class NeighborEmb(MessagePassing):
    r"""Initialize node features based on neighboring nodes."""

    def __init__(self, hid_dim, in_hidden_channels=5):
        super(NeighborEmb, self).__init__(aggr="add")
        self.embedding = nn.Linear(in_hidden_channels, hid_dim)
        self.hid_dim = hid_dim
        self.ln_emb = nn.LayerNorm(hid_dim, elementwise_affine=False)

    def forward(self, z, s, edge_index, embs):
        s_neighbors = self.ln_emb(self.embedding(z))
        s_neighbors = self.propagate(edge_index, x=s_neighbors, norm=embs)

        s = s + s_neighbors
        return s

    def message(self, x_j, norm):
        return norm.view(-1, self.hid_dim) * x_j


class CFConvS2V(MessagePassing):
    r"""Scalar to vector."""

    def __init__(self, hid_dim: int):
        super(CFConvS2V, self).__init__(aggr="add")
        self.hid_dim = hid_dim
        self.lin1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim, elementwise_affine=False),
            nn.SiLU(),
        )

    def forward(self, s, v, edge_index, emb):
        """_summary_

        Args:
            s (_type_): _description_, [n_atom, n_z, n_embed]
            v (_type_): _description_, [n_edge, n_pos, n_embed]
            edge_index (_type_): _description_, [2, n_edge]
            emb (_type_): _description_, [n_edge, n_embed]

        Returns:
            _type_: _description_
        """
        s = self.lin1(s)
        emb = emb.unsqueeze(1) * v

        v = self.propagate(edge_index, x=s, norm=emb)
        return v.view(-1, 3, self.hid_dim)

    def message(self, x_j, norm):
        x_j = x_j.unsqueeze(1)
        a = norm.view(-1, 3, self.hid_dim) * x_j
        return a.view(-1, 3 * self.hid_dim)


class GCLMessage(nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_radial,
        act_fn: str = "swish",
        legacy: bool = False,
    ):
        super().__init__()
        self.edge_mlp = MLP(
            in_dim=hidden_channels * 2 + 3 * hidden_channels + num_radial,
            out_dims=[hidden_channels, hidden_channels],
            activation=act_fn,
        )
        self.node_mlp = MLP(
            in_dim=(hidden_channels) + hidden_channels,
            out_dims=[hidden_channels, hidden_channels],
            activation=act_fn,
            last_layer_no_activation=True if legacy else False,
        )
        self.edge_out_trans = MLP(
            in_dim=hidden_channels,
            out_dims=[3 * hidden_channels + num_radial],
            activation=act_fn,
        )
        self.att_mlp = MLP((hidden_channels), [1], activation=act_fn)
        self.x_layernorm = nn.LayerNorm(hidden_channels)
        self.x_layernorm.reset_parameters()

    def forward(self, x, edge_index, weight):
        xh = self.x_layernorm(x)
        edgeh = weight

        ii, jj = edge_index
        m_ij = self.edge_message(xh[ii], xh[jj], edgeh)
        xh = self.node_message(xh, edge_index, m_ij)
        edgeh = edgeh + self.edge_out_trans(m_ij)
        return xh, edgeh

    def edge_message(self, xh_i, xh_j, edgeh):
        m_ij = self.edge_mlp(torch.cat([xh_i, xh_j, edgeh], dim=1))
        m_ij = m_ij * self.att_mlp(m_ij)
        return m_ij

    def node_message(self, xh, edge_index, m_ij):
        ii, jj = edge_index
        agg = unsorted_segment_sum(
            m_ij,
            ii,
            num_segments=xh.size(0),
            normalization_factor=1,
            aggregation_method="mean",
        )
        agg = torch.cat([xh, agg], dim=1)
        xh = xh + self.node_mlp(agg)
        return xh


class EquiMessage(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_radial,
        reflect_equiv,
    ):
        super(EquiMessage, self).__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels
        self.num_radial = num_radial
        self.reflect_equiv = reflect_equiv
        self.dir_proj = nn.Sequential(
            nn.Linear(
                3 * self.hidden_channels + self.num_radial,
                self.hidden_channels * 3,
            ),
            nn.SiLU(inplace=True),
            nn.Linear(
                self.hidden_channels * 3,
                self.hidden_channels * 3,
            ),
        )

        self.x_proj = nn.Sequential(
            nn.Linear(
                hidden_channels,
                hidden_channels,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_channels,
                hidden_channels * 3,
                bias=False,
            ),
        )
        self.rbf_proj = nn.Linear(
            num_radial,
            hidden_channels * 3,
            bias=False,
        )

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        # self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        # self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        # self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()

    def forward(self, x, vec, edge_index, edge_rbf, weight, edge_vector, edge_cross):
        xh = self.x_proj(self.x_layernorm(x))

        rbfh = self.rbf_proj(edge_rbf)
        weight = self.dir_proj(weight)
        rbfh = rbfh * weight
        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor, edge_cross: Tensor)
        dx, dvec = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            edge_cross=edge_cross,
            size=None,
        )

        return dx, dvec

    def message(self, xh_j, xh_i, vec_j, rbfh_ij, r_ij, edge_cross):
        x, xh2, xh3 = torch.split((xh_j + xh_i) * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3

        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        if not self.reflect_equiv:  # Added by Chenru: for reflection taking effects.
            vec = vec + x.unsqueeze(1) * edge_cross.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class EquiUpdate(nn.Module):
    def __init__(self, hidden_channels, reflect_equiv: bool = True):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3, bias=False),
        )

        self.lin3 = nn.Sequential(
            nn.Linear(3, 48),
            nn.SiLU(inplace=True),
            nn.Linear(48, 8),
            nn.SiLU(inplace=True),
            nn.Linear(8, 1),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.reflect_equiv = reflect_equiv

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        # self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        # self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec, nodeframe):
        vec = self.vec_proj(vec)
        vec1, vec2 = torch.split(vec, self.hidden_channels, dim=-1)

        scalrization = torch.sum(vec1.unsqueeze(2) * nodeframe.unsqueeze(-1), dim=1)

        if self.reflect_equiv:
            scalrization[:, 1, :] = torch.abs(scalrization[:, 1, :].clone())
        scalar = (self.lin3(torch.permute(scalrization, (0, 2, 1)))).squeeze(-1)

        vec_dot = (vec1 * vec2).sum(dim=1)
        vec_dot = vec_dot * self.inv_sqrt_h

        x_vec_h = self.xvec_proj(torch.cat([x, scalar], dim=-1))
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)

        dx = xvec1 + xvec2 + vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec2

        return dx, dvec


class _EquiUpdate(nn.Module):
    def __init__(self, hidden_channels, reflect_equiv: bool = True):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.vec_proj2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.lin3 = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 8),
            # nn.BatchNorm1d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(8, 1),
        )

        self.lin4 = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 8),
            # nn.BatchNorm1d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(8, 1),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec, nodeframe):
        vec = self.vec_proj(vec)
        vec1, vec2 = torch.split(vec, self.hidden_channels, dim=-1)
        scalrization = torch.sum(vec1.unsqueeze(2) * nodeframe.unsqueeze(-1), dim=1)
        scalrization[:, 1, :] = torch.abs(scalrization[:, 1, :].clone())

        scalar = torch.sqrt(torch.sum(vec1**2, dim=-2))
        scalrization1 = torch.sum(vec2.unsqueeze(2) * nodeframe.unsqueeze(-1), dim=1)
        scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())

        vec_dot = self.lin4(
            torch.permute(torch.cat([scalrization, scalrization1], dim=-2), (0, 2, 1))
        ).squeeze(-1)
        vec_dot = vec_dot * self.inv_sqrt_h

        # NOTE: Can't use torch.norm because the gradient is NaN for input = 0.
        # Add an epsilon offset to make sure sqrt is always positive.

        x_vec_h = self.xvec_proj(torch.cat([x, scalar], dim=-1))
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)

        dx = xvec1 + xvec2 * vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec2

        return dx, dvec


class vector(MessagePassing):
    def __init__(self):
        super(vector, self).__init__(aggr="mean")

    def forward(self, vector, edge_index):
        v = self.propagate(edge_index, x=vector)

        return v


def nn_vector(dist: Tensor, edge_index: Tensor, pos: Tensor):
    r"""Added by Chenru: Getting the nearest neighbor position to construct nodeframe.

    Args:
        dist (Tensor): (n_edge)
        edge_index (Tensor): (2, n_edge)
        pos (Tensor): (n_atom, 3)
    Returns:
        Tensor: (n_atom, 3)
    """
    ii, jj = edge_index
    vec = []
    pairs = {}
    for n in range(pos.size(0)):
        if n not in pairs:
            inds = torch.where(ii == n)[0]
            if not len(inds):
                nn_j = n
            else:
                min_ind = torch.argmin(dist[inds])
                nn_j = jj[inds][min_ind].item()
            pairs.update({nn_j: n})
        else:
            nn_j = pairs[n]
        vec.append(pos[nn_j])
    vec = torch.stack(vec)

    # vec = torch.rand_like(pos)  # to test assert_rot_equiv is working

    return vec


def assert_rot_equiv(func: Callable, dist: Tensor, edge_index: Tensor, pos: Tensor):
    r"""Added by Chenru: test a func for constructing y1 is equivariant.

    Args:
        func (Callable): _description_
        dist (Tensor): _description_
        edge_index (Tensor): _description_
        pos (Tensor): _description_
    """
    theta = 0.4
    alpha = 0.9
    rot_x = torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float64,
    )
    rot_y = torch.tensor(
        [
            [np.cos(alpha), 0, np.sin(alpha)],
            [0, 1, 0],
            [-np.sin(alpha), 0, np.cos(alpha)],
        ],
        dtype=torch.float64,
    )
    rot = torch.matmul(rot_y, rot_x).double()
    y1 = func(dist, edge_index, pos)
    pos_new = torch.matmul(pos, rot).double()
    y1_new = func(dist, edge_index, pos_new)
    assert torch.allclose(
        torch.matmul(y1, rot).double(),
        y1_new,
    )


class EquiOutput(nn.Module):
    def __init__(self, hidden_channels, out_channels=1, single_layer_output=True):
        super().__init__()
        self.hidden_channels = hidden_channels

        if single_layer_output:
            self.output_network = nn.ModuleList(
                [
                    GatedEquivariantBlock(hidden_channels, out_channels),
                ]
            )
        else:
            self.output_network = nn.ModuleList(
                [
                    GatedEquivariantBlock(hidden_channels, hidden_channels // 2),
                    GatedEquivariantBlock(hidden_channels // 2, out_channels),
                ]
            )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return x, vec.squeeze()


class GatedEquivariantBlock(nn.Module):
    r"""
    Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra.

    Borrowed from TorchMD-Net
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )

        self.act = nn.Identity()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = self.act(x)
        # v = torch.mean(v, dim=-1)
        return x, v


class LEFTNet(torch.nn.Module):
    r"""
    LEFTNet

    Args:
        pos_require_grad (bool, optional): If set to :obj:`True`, will require to take derivative of model output with respect to the atomic positions. (default: :obj:`False`)
        cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
        num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
        hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
        num_radial (int, optional): Number of radial basis functions. (default: :obj:`96`)
        y_mean (float, optional): Mean value of the labels of training data. (default: :obj:`0`)
        y_std (float, optional): Standard deviation of the labels of training data. (default: :obj:`1`)

    """

    def __init__(
        self,
        pos_require_grad=False,
        cutoff=10.0,
        num_layers=4,
        hidden_channels=128,
        num_radial=96,
        in_hidden_channels: int = 8,
        reflect_equiv: bool = True,
        legacy: bool = True,
        update: bool = True,
        pos_grad: bool = False,
        single_layer_output: bool = True,
        for_conf: bool = False,
        ff: bool = False,
        object_aware: bool = True,
        **kwargs,
    ):
        super(LEFTNet, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.pos_require_grad = pos_require_grad
        self.reflect_equiv = reflect_equiv
        self.legacy = legacy
        self.update = update
        self.pos_grad = pos_grad
        self.for_conf = for_conf
        self.ff = ff
        self.object_aware = object_aware

        self.embedding = nn.Linear(in_hidden_channels, hidden_channels)
        self.embedding_out = nn.Linear(hidden_channels, in_hidden_channels)
        self.radial_emb = RBFEmb(num_radial, self.cutoff)
        self.neighbor_emb = NeighborEmb(hidden_channels, in_hidden_channels)
        self.s2v = CFConvS2V(hidden_channels)

        self.radial_lin = nn.Sequential(
            nn.Linear(num_radial, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.lin3 = nn.Sequential(
            nn.Linear(3, hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels // 4, 1),
        )
        self.pos_expansion = MLP(
            in_dim=3,
            out_dims=[hidden_channels // 2, hidden_channels],
            activation="swish",
            last_layer_no_activation=True,
            bias=False,
        )
        if self.legacy:
            self.distance_embedding = MLP(
                in_dim=num_radial,
                out_dims=[hidden_channels // 2, hidden_channels],
                activation="swish",
                bias=False,
            )
        if self.pos_grad:
            self.dynamic_mlp_modules = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_channels // 2, 3),
            )

        self.gcl_layers = nn.ModuleList()
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.gcl_layers.append(
                GCLMessage(hidden_channels, num_radial, legacy=legacy)
            )
            self.message_layers.append(
                EquiMessage(hidden_channels, num_radial, reflect_equiv).jittable()
            )
            self.update_layers.append(EquiUpdate(hidden_channels, reflect_equiv))

        self.last_layer = nn.Linear(hidden_channels, 1)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.out_pos = EquiOutput(
            hidden_channels,
            out_channels=1,
            single_layer_output=single_layer_output,
        )

        # for node-wise frame
        self.vec = vector()

        self.reset_parameters()

    def reset_parameters(self):
        self.radial_emb.reset_parameters()

    def scalarization(self, pos, edge_index):
        i, j = edge_index
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        coord_diff = pos[i] - pos[j]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
        coord_cross = torch.cross(pos[i], pos[j])
        norm = torch.sqrt(radial) + EPS
        coord_diff = coord_diff / norm
        cross_norm = (torch.sqrt(torch.sum((coord_cross) ** 2, 1).unsqueeze(1))) + EPS
        coord_cross = coord_cross / cross_norm
        coord_vertical = torch.cross(coord_diff, coord_cross)

        return dist, coord_diff, coord_cross, coord_vertical

    @staticmethod
    def assemble_nodemask(edge_index: Tensor, pos: Tensor):
        node_mask = torch.zeros(pos.size(0), device=pos.device)
        node_mask[:] = -1
        _i, _j = edge_index
        _ind = 0
        for center in range(pos.size(0)):
            if node_mask[center] > -1:
                continue
            _connected = _j[torch.where(_i == center)]
            _connected = torch.concat(
                [_connected, torch.tensor([center], device=pos.device)]
            )
            node_mask[_connected] = _ind
            _ind += 1
        return node_mask

    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
        edge_mask: Optional[Tensor] = None,
        update_coords_mask: Optional[Tensor] = None,
        subgraph_mask: Optional[Tensor] = None,
    ):
        # if self.pos_require_grad:
        #     pos.requires_grad_()

        if not self.object_aware:
            subgraph_mask = None

        i, j = edge_index

        # embed z, assuming last column is atom number
        z_emb = self.embedding(h)

        i, j = edge_index
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        inner_subgraph_mask = torch.zeros(edge_index.size(1), 1, device=dist.device)
        inner_subgraph_mask[torch.where(dist < self.cutoff)[0]] = 1

        all_edge_masks = inner_subgraph_mask
        if subgraph_mask is not None:
            all_edge_masks = all_edge_masks * subgraph_mask

        edge_index_w_cutoff = edge_index.T[torch.where(all_edge_masks > 0)[0]].T
        node_mask_w_cutoff = self.assemble_nodemask(
            edge_index=edge_index_w_cutoff, pos=pos
        )

        pos_frame = pos.clone()
        pos_frame = remove_mean_batch(pos_frame, node_mask_w_cutoff.long())

        # bulid edge-wise frame and scalarization vector features for edge update
        dist, coord_diff, coord_cross, coord_vertical = self.scalarization(
            pos_frame, edge_index
        )

        dist = dist * all_edge_masks.squeeze(-1)
        coord_diff = coord_diff * all_edge_masks
        coord_cross = coord_cross * all_edge_masks
        coord_vertical = coord_vertical * all_edge_masks

        frame = torch.cat(
            (
                coord_diff.unsqueeze(-1),
                coord_cross.unsqueeze(-1),
                coord_vertical.unsqueeze(-1),
            ),
            dim=-1,
        )
        radial_emb = self.radial_emb(dist)
        radial_emb = radial_emb * all_edge_masks

        f = self.radial_lin(radial_emb)
        rbounds = 0.5 * (torch.cos(dist * pi / self.cutoff) + 1.0)
        f = rbounds.unsqueeze(-1) * f

        # init node features
        s = self.neighbor_emb(h, z_emb, edge_index, f)

        NE1 = self.s2v(s, coord_diff.unsqueeze(-1), edge_index, f)
        scalrization1 = torch.sum(NE1[i].unsqueeze(2) * frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(NE1[j].unsqueeze(2) * frame.unsqueeze(-1), dim=1)
        if self.reflect_equiv:
            scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
            scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())

        scalar3 = (
            self.lin3(torch.permute(scalrization1, (0, 2, 1)))
            + torch.permute(scalrization1, (0, 2, 1))[:, :, 0].unsqueeze(2)
        ).squeeze(-1)
        scalar4 = (
            self.lin3(torch.permute(scalrization2, (0, 2, 1)))
            + torch.permute(scalrization2, (0, 2, 1))[:, :, 0].unsqueeze(2)
        ).squeeze(-1)
        edgeweight = torch.cat((scalar3, scalar4), dim=-1) * rbounds.unsqueeze(-1)
        edgeweight = torch.cat((edgeweight, f), dim=-1)
        # add distance embedding
        edgeweight = torch.cat((edgeweight, radial_emb), dim=-1)

        # bulid node-wise frame for node-update
        a = pos_frame
        if self.legacy:
            b = self.vec(pos_frame, edge_index)
        else:
            # Added by Chenru: for new implementation of constructing node frame.
            eff_edge_ij = torch.where(all_edge_masks.squeeze(-1) == 1)[0]
            eff_edge_index = edge_index[:, eff_edge_ij]
            eff_dist = dist[eff_edge_ij]
            b = nn_vector(eff_dist, eff_edge_index, pos_frame)
        # assert_rot_equiv(nn_vector, dist_pad, edge_index, pos)  # for debugging

        x1 = (a - b) / ((torch.sqrt(torch.sum((a - b) ** 2, 1).unsqueeze(1))) + EPS)
        y1 = torch.cross(a, b)
        normy = (torch.sqrt(torch.sum(y1**2, 1).unsqueeze(1))) + EPS
        y1 = y1 / normy
        # assert torch.trace(torch.matmul(x1, torch.transpose(y1, 0, 1))) < EPS  # for debugging

        z1 = torch.cross(x1, y1)
        nodeframe = torch.cat(
            (x1.unsqueeze(-1), y1.unsqueeze(-1), z1.unsqueeze(-1)), dim=-1
        )

        pos_prjt = torch.sum(pos_frame.unsqueeze(-1) * nodeframe, dim=1)

        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device)
        gradient = torch.zeros(s.size(0), 3, device=s.device)
        for i in range(self.num_layers):
            # Added by Chenru: for letting multiple objects message passing.
            if self.legacy or i == 0:
                s = s + self.pos_expansion(pos_prjt)
            s, edgeweight = self.gcl_layers[i](
                s,
                edge_index,
                edgeweight,
            )

            dx, dvec = self.message_layers[i](
                s,
                vec,
                edge_index,
                radial_emb,
                edgeweight,
                coord_diff,
                coord_cross,
            )
            s = s + dx
            vec = vec + dvec
            s = s * self.inv_sqrt_2

            if self.update:
                dx, dvec = self.update_layers[i](s, vec, nodeframe)
                s = s + dx
                vec = vec + dvec

            if self.pos_grad:
                dynamic_coff = self.dynamic_mlp_modules(s)  # (node, 3)
                basis_mix = (
                    dynamic_coff[:, :1] * x1
                    + dynamic_coff[:, 1:2] * y1
                    + dynamic_coff[:, 2:3] * z1
                )
                gradient = gradient + basis_mix / self.num_layers

        if self.for_conf:
            return s

        _, dpos = self.out_pos(s, vec)

        if update_coords_mask is not None:
            dpos = update_coords_mask * dpos
        pos = pos + dpos + gradient

        if self.ff:
            return s, dpos

        h = self.embedding_out(s)
        if node_mask is not None:
            h = h * node_mask
        edge_attr = None
        return h, pos, edge_attr
