"""EGNN model"""
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from .block import EquivariantBlock, SinusoidsEmbeddingNew
from .util_funcs import coord2diff, symmetrize_edge, get_ji_bond_index


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf: int = 8,
        in_edge_nf: int = 2,
        hidden_nf: int = 256,
        edge_hidden_nf: int = 32,
        act_fn: str = "swish",
        n_layers: int = 3,
        attention: int = False,
        out_node_nf: Optional[int] = None,
        tanh: bool = False,
        coords_range: float = 15.0,
        norm_constant: float = 1.0,
        inv_sublayers: int = 2,
        sin_embedding: bool = False,
        normalization_factor: float = 100.0,
        aggregation_method: str = "sum",
        reflect_equiv: bool = True,
    ):
        r"""_summary_

        Args:
            in_node_nf (int): number of input node feature. Defaults to 8.
            in_edge_nf (int): number of input edge feature. Defaults to 2.
            hidden_nf (int): number of hidden units. Defaults to 256.
            act_fn (str): activation function. Defaults to "swish".
            n_layers (int): number of equivariant update block. Defaults to 3.
            attention (int): whether to use self attention. Defaults to False.
            out_node_nf (Optional[int]): number of output node features.
                Defaults to None to set the same as in_node_nf
            coords_range (float): range factor, only used in tanh = True.
                Defaults to 15.0.
            norm_constant (float): distance normalizating factor. Defaults to 1.0.
            inv_sublayers (int): number of GCL in an equivariant update block.
                Defaults to 2.
            sin_embedding (Optional[nn.Module]): whether to use edge distance embedding.
                Defaults to None.
            normalization_factor (float): distance normalization used in coord2diff.
                Defaults to 1.0.
            aggregation_method (str): aggregation options in scattering.
                Defaults to "sum".
            reflect_equiv (bool): whether to ignore reflection.
                Defaults to True.
        """
        super().__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.edge_hidden_nf = edge_hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflect_equiv = reflect_equiv

        edge_feat_nf = in_edge_nf
        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            self.dist_dim = self.sin_embedding.dim
        else:
            self.sin_embedding = None
            self.dist_dim = 1

        self.edge_feat_nf = edge_feat_nf + self.dist_dim

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        self.edge_embedding = nn.Linear(
            self.edge_feat_nf, self.hidden_nf - self.dist_dim
        )
        self.edge_embedding_out = nn.Linear(
            self.hidden_nf - self.dist_dim, self.edge_feat_nf
        )
        for i in range(0, n_layers):
            self.add_module(
                "e_block_%d" % i,
                EquivariantBlock(
                    hidden_nf,
                    edge_feat_nf=edge_feat_nf,
                    act_fn=act_fn,
                    n_layers=inv_sublayers,
                    attention=attention,
                    tanh=tanh,
                    coords_range=coords_range,
                    norm_constant=norm_constant,
                    sin_embedding=self.sin_embedding,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method,
                    reflect_equiv=reflect_equiv,
                ),
            )

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
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""

        Args:
            h (Tensor): [n_nodes, n_hidden], node features.
            pos (Tensor): [n_nodes, n_dim (3 in 3D space)], position tensor.
            edge_index (Tensor): [2, n_edge], edge index {ij}
            edge_attr (Optional[Tensor]): [n_edge, edge_feature_dim]. edge attributes.
                Defaults to None.
            node_mask (Optional[Tensor]): [n_node, 1], mask for node updates.
                Defaults to None.
            edge_mask (Optional[Tensor]): [n_edge, 1], mask for edge updates.
                Defaults to None.
            update_coords_mask (Optional[Tensor]): [n_node, 1], mask for position updates.
                Defaults to None.
            subgraph_mask (Optional[Tensor]): n_edge, 1], mask for positions aggregations.
                The idea is keep subgraph (i.e., fragment) level equivariance.
                Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: updated h, pos, edge_attr
        """
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(pos, edge_index)
        if subgraph_mask is not None:
            distances = distances * subgraph_mask

        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        if edge_attr is None or edge_attr.size(-1) == 0:
            edge_attr = distances
        else:
            edge_attr = torch.concat([distances, edge_attr], dim=-1)
        edge_attr = self.edge_embedding(edge_attr)
        h = self.embedding(h)
        # edge_index_ji = get_ji_bond_index(edge_index)
        # edge_attr = symmetrize_edge(edge_attr, edge_index_ji)

        for i in range(0, self.n_layers):
            h, pos, edge_attr = self._modules["e_block_%d" % i](
                h,
                pos,
                edge_index,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
                update_coords_mask=update_coords_mask,
                subgraph_mask=subgraph_mask,
            )

            # edge_attr = symmetrize_edge(edge_attr, edge_index_ji)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        edge_attr = self.edge_embedding_out(edge_attr)

        if node_mask is not None:
            h = h * node_mask
        if edge_mask is not None:
            edge_attr = edge_attr * edge_mask
        return h, pos, edge_attr
