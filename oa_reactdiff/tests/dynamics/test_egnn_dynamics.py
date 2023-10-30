"""Test model forward pass and equivariance."""
import unittest
from typing import List, Optional

import torch
from torch import Tensor, tensor, nn
from pytorch_lightning import seed_everything

from oa_reactdiff.model import LEFTNet
from oa_reactdiff.dynamics import EGNNDynamics, Confidence
from oa_reactdiff.utils import (
    get_n_frag_switch,
    get_mask_for_frag,
    get_edges_index,
)

seed_everything(0, workers=True)


def init_weights(m):
    r"""Weight initialization for all MLP.

    Args:
        m: a nn.Module
    """
    if isinstance(m, nn.Linear):
        gain = 0.5
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -gain, gain)


egnn_config = dict(
    in_node_nf=8,
    in_edge_nf=2,
    hidden_nf=2,
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
    normalization_factor=100.0,
    aggregation_method="sum",
)
leftnet_config = dict(
    pos_require_grad=False,
    cutoff=5.0,
    num_layers=2,
    hidden_channels=32,
    num_radial=8,
    in_node_nf=8,
)

node_nfs: List[int] = [4, 5, 6]
edge_nf: int = 3
condition_nf: int = 3
fragment_names: List[str] = ["inorg_node", "org_edge", "org_node"]
pos_dim: int = 3
update_pocket_coords: bool = True
condition_time: bool = True
edge_cutoff: Optional[float] = None


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.egnn_dynamics = EGNNDynamics(
            model_config=egnn_config,
            node_nfs=node_nfs,
            edge_nf=edge_nf,
            condition_nf=condition_nf,
            fragment_names=fragment_names,
            pos_dim=pos_dim,
            update_pocket_coords=update_pocket_coords,
            condition_time=condition_time,
            edge_cutoff=edge_cutoff,
        )
        cls.egnn_dynamics.model.apply(init_weights)

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

        cls.n_samples = 2
        cls.fragments_nodes = [
            torch.tensor([2, 0]),
            torch.tensor([2, 3]),
            torch.tensor([1, 2]),
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
        cls.edge_attr = (
            torch.rand(
                cls.edge_index.size(1),
                edge_nf,
            )
            if edge_nf > 0
            else None
        )

        cls.confidence = Confidence(
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

    def test_basic_elements(self):
        self.assertTrue(
            torch.allclose(self.n_frag_switch, tensor([0, 0, 1, 1, 1, 1, 1, 2, 2, 2]))
        )
        self.assertTrue(
            torch.allclose(self.combined_mask, tensor([0, 0, 0, 0, 1, 1, 1, 0, 1, 1]))
        )
        self.assertTrue(self.edge_index.shape == (2, 40))
        for ii, n_nodes in enumerate(self.fragments_nodes):
            self.assertTrue(
                self.xh[ii].shape == (torch.sum(n_nodes).item(), node_nfs[ii])
            )

    def test_forward_dynamics(self):
        for dynamics in self.dynamics:
            _xh, _edge_attr = dynamics.forward(
                self.xh,
                self.edge_index,
                self.t,
                self.conditions,
                self.n_frag_switch,
                self.combined_mask,
                edge_attr=self.edge_attr,
            )
            for ii, _ in enumerate(self.fragments_nodes):
                self.assertTrue(self.xh[ii].shape == _xh[ii].shape)
            if _edge_attr is not None:
                self.assertTrue(self.edge_attr.shape == _edge_attr.shape)

            _, _ = dynamics.forward(
                _xh,
                self.edge_index,
                self.t,
                self.conditions,
                self.n_frag_switch,
                self.combined_mask,
                edge_attr=_edge_attr,
            )

    def test_condition_functioning(self):
        for dynamics in self.dynamics:
            _xh, _edge_attr = dynamics.forward(
                self.xh,
                self.edge_index,
                self.t,
                self.conditions,
                self.n_frag_switch,
                self.combined_mask,
                edge_attr=self.edge_attr,
            )
            _xh_t, _edge_attr_t = dynamics.forward(
                self.xh,
                self.edge_index,
                torch.tensor([314]),
                self.conditions,
                self.n_frag_switch,
                self.combined_mask,
                edge_attr=self.edge_attr,
            )

            for ii, _ in enumerate(self.fragments_nodes):
                self.assertFalse(
                    torch.allclose(
                        _xh[ii],
                        _xh_t[ii],
                        rtol=1e-3,
                    )
                )

            _xh_condition, _edge_attr_condition = dynamics.forward(
                self.xh,
                self.edge_index,
                self.t,
                torch.rand(self.n_samples, condition_nf),
                self.n_frag_switch,
                self.combined_mask,
                edge_attr=self.edge_attr,
            )

            for ii, _ in enumerate(self.fragments_nodes):
                self.assertFalse(
                    torch.allclose(
                        _xh[ii],
                        _xh_condition[ii],
                        rtol=1e-4,
                    )
                )

    def test_edge_adjustment(self):
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        h = torch.rand(4, egnn_config["in_node_nf"])
        pos = torch.rand(4, 3)
        edge_attr = torch.rand(edge_index.size(1), egnn_config["in_edge_nf"])

        edge_index_new = torch.tensor(
            [[1, 2, 0, 3, 1], [2, 1, 1, 1, 3]], dtype=torch.long
        )
        self.egnn_dynamics.adjust_edge_attr_on_new_eij(
            edge_index,
            edge_attr,
            edge_index_new,
        )

    def test_forward_confidence(self):
        conf = self.confidence._forward(
            self.xh,
            self.edge_index,
            tensor([0]),
            self.conditions,
            self.n_frag_switch,
            self.combined_mask,
            edge_attr=self.edge_attr,
        )
        assert conf.size(0) == self.n_samples
