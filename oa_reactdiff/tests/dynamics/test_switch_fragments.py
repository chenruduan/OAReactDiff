"""Test model forward pass and equivariance."""
import unittest
from typing import List, Optional

import torch
from torch import Tensor, tensor, nn
from pytorch_lightning import seed_everything

from oa_reactdiff.model import LEFTNet
from oa_reactdiff.dynamics import EGNNDynamics
from oa_reactdiff.utils import (
    get_n_frag_switch,
    get_mask_for_frag,
    get_edges_index,
)

seed_everything(0, workers=True)

model_config = dict(
    in_node_nf=8,
    in_edge_nf=0,
    hidden_nf=64,
    edge_hidden_nf=3,
    act_fn="swish",
    n_layers=6,
    attention=True,
    out_node_nf=None,
    tanh=False,
    coords_range=15.0,
    norm_constant=1.0,
    inv_sublayers=2,
    sin_embedding=False,
    normalization_factor=1.0,
    aggregation_method="mean",
)
leftnet_config = dict(
    pos_require_grad=False,
    cutoff=5.0,
    num_layers=2,
    hidden_channels=32,
    num_radial=8,
    in_node_nf=8,
)

node_nfs: List[int] = [5] * 2
edge_nf: int = 4
condition_nf: int = 3
fragment_names: List[str] = ["A", "B"]
pos_dim: int = 3
update_pocket_coords: bool = True
condition_time: bool = True
edge_cutoff: Optional[float] = None


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.egnn_dynamics = EGNNDynamics(
            model_config=model_config,
            node_nfs=node_nfs,
            edge_nf=edge_nf,
            condition_nf=condition_nf,
            fragment_names=fragment_names,
            pos_dim=pos_dim,
            update_pocket_coords=update_pocket_coords,
            condition_time=condition_time,
            edge_cutoff=edge_cutoff,
        )
        cls.leftnet_dynamics = EGNNDynamics(
            model_config=leftnet_config,
            node_nfs=node_nfs,
            edge_nf=edge_nf,
            condition_nf=condition_nf,
            fragment_names=fragment_names,
            pos_dim=pos_dim,
            update_pocket_coords=update_pocket_coords,
            condition_time=condition_time,
            edge_cutoff=edge_cutoff,
            model=LEFTNet,
        )
        cls.dynamics = [cls.egnn_dynamics, cls.leftnet_dynamics]

        cls.n_samples = 1
        cls.fragments_nodes = [
            torch.tensor(
                [
                    4,
                ]
            ),
            torch.tensor(
                [
                    5,
                ]
            ),
        ]
        cls.fragments_masks = [
            get_mask_for_frag(natm_nodes) for natm_nodes in cls.fragments_nodes
        ]
        cls.conditions = torch.rand(cls.n_samples, condition_nf)

        cls.n_frag_switch = get_n_frag_switch(cls.fragments_nodes)

        cls.combined_mask = torch.cat(cls.fragments_masks)
        cls.edge_index = get_edges_index(cls.combined_mask, remove_self_edge=True)

        cls.xh = [
            torch.rand(torch.sum(cls.fragments_nodes[ii]), node_nfs[ii])
            for ii in range(len(node_nfs))
        ]
        cls.t = torch.tensor([0.314])

    def test_switch_fragments(self):
        """Switch the location of fragments should result in different outputs."""
        for dynamics in self.dynamics:
            _xh, _edge_attr = dynamics.forward(
                self.xh,
                self.edge_index,
                self.t,
                self.conditions,
                self.n_frag_switch,
                self.combined_mask,
                edge_attr=None,
            )

            xh = [self.xh[1], self.xh[0]]
            fragments_nodes = [
                self.fragments_nodes[1],
                self.fragments_nodes[0],
            ]
            fragments_masks = [
                get_mask_for_frag(natm_nodes) for natm_nodes in fragments_nodes
            ]
            n_frag_switch = get_n_frag_switch(fragments_nodes)

            combined_mask = torch.cat(fragments_masks)
            edge_index = get_edges_index(combined_mask, remove_self_edge=True)
            _xh_switch, _edge_attr_switch = dynamics.forward(
                xh,
                edge_index,
                self.t,
                self.conditions,
                n_frag_switch,
                combined_mask,
                edge_attr=None,
            )

            print(_xh[0])
            print(_xh_switch[1])
            self.assertFalse(
                torch.allclose(
                    _xh[0],
                    _xh_switch[1],
                    rtol=1e-6,
                )
            )

    def test_switch_fragments_same_encoding(self):
        """
        If the encoder and decoder are the same for different fragments,
        switch the location of fragments should result in *same* outputs.
        """
        for dynamics in self.dynamics:
            dynamics.encoders[1] = dynamics.encoders[0]
            dynamics.decoders[1] = dynamics.decoders[0]
            _xh, _edge_attr = dynamics.forward(
                self.xh,
                self.edge_index,
                self.t,
                self.conditions,
                self.n_frag_switch,
                self.combined_mask,
                edge_attr=None,
            )

            xh = [self.xh[1], self.xh[0]]
            fragments_nodes = [
                self.fragments_nodes[1],
                self.fragments_nodes[0],
            ]
            fragments_masks = [
                get_mask_for_frag(natm_nodes) for natm_nodes in fragments_nodes
            ]
            n_frag_switch = get_n_frag_switch(fragments_nodes)

            combined_mask = torch.cat(fragments_masks)
            edge_index = get_edges_index(combined_mask, remove_self_edge=True)
            _xh_switch, _edge_attr_switch = dynamics.forward(
                xh,
                edge_index,
                self.t,
                self.conditions,
                n_frag_switch,
                combined_mask,
                edge_attr=None,
            )

            print(_xh[0])
            print(_xh_switch[1])
            self.assertTrue(
                torch.allclose(
                    _xh[0],
                    _xh_switch[1],
                    rtol=1e-6,
                )
            )
