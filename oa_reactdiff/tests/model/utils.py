"""Ultility functions used in test cases."""
import torch
from torch import nn


egnn_config = dict(
    in_node_nf=8,
    in_edge_nf=5,
    hidden_nf=64,
    edge_hidden_nf=64,
    act_fn="swish",
    n_layers=6,
    attention=True,
    out_node_nf=None,
    tanh=True,
    coords_range=15.0,
    norm_constant=1.0,
    inv_sublayers=2,
    sin_embedding=False,
    normalization_factor=1.0,
    aggregation_method="mean",
)

left_config = dict(
    pos_require_grad=False,
    cutoff=20.0,
    num_layers=6,
    hidden_channels=32,
    num_radial=32,
    in_node_nf=8,
    reflect_equiv=True,
)


def tensor_relative_diff(x1, x2):
    return torch.max(torch.abs(x1 - x2) / (x1 + x2 + 1e-6) * 2)


def init_weights(m):
    r"""Weight initialization for all MLP.

    Args:
        m: a nn.Module
    """
    if isinstance(m, nn.Linear):
        gain = 1.0
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -gain, gain)


def generate_full_eij(n_atom: int):
    r"""Get fully-connected graphs for n_atoms."""
    edge_index = []
    for ii in range(n_atom):
        for jj in range(n_atom):
            if ii != jj:
                edge_index.append([ii, jj])
    return torch.transpose(torch.Tensor(edge_index), 1, 0).long()


def get_cut_graph_mask(edge_index, n_cut):
    r"""Get mask for a graph cut at n_cut, with ij representing cross-subgraph edgs being 0."""
    ind_sum = torch.where(edge_index < n_cut, 1, 0).sum(dim=0)
    subgraph_mask = torch.zeros(edge_index.size(1)).long()
    subgraph_mask[ind_sum == 2] = 1
    subgraph_mask[ind_sum == 0] = 1
    subgraph_mask = subgraph_mask[:, None]
    return subgraph_mask
