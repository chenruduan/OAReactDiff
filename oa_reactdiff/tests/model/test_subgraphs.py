"""Test for subgraph_mask, where the position updates are killed for subgraphs, but not the h and edge updates"""
import unittest
import torch
from torch import nn
import numpy as np
from pytorch_lightning import seed_everything

from oa_reactdiff.model import EGNN, LEFTNet
from .utils import (
    tensor_relative_diff,
    egnn_config,
    init_weights,
    generate_full_eij,
    get_cut_graph_mask,
    left_config,
)

default_float = torch.float64
torch.set_default_dtype(default_float)
EPS = 1e-6
LARGE_EPS = 1e-4

theta = 0.9
alpha = 0.4
seed_everything(1234, workers=True)


def com(x):
    return x - torch.mean(x, dim=0)


class TestSubGraphs(unittest.TestCase):
    """Test model behavior when we have two fragments in a system."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.egnn = EGNN(**egnn_config)
        cls.leftnet = LEFTNet(**left_config)
        n1, n2 = 4, 9
        ntot = n1 + n2
        cls.n1 = n1
        cls.edge_index = generate_full_eij(ntot)
        cls.h = torch.rand(ntot, egnn_config["in_node_nf"])
        cls.pos = torch.cat(
            [
                com(torch.rand(n1, 3)),
                com(torch.rand(n2, 3)),
            ],
            dim=0,
        )
        cls.edge_attr = torch.rand(cls.edge_index.size(1), egnn_config["in_edge_nf"])
        cls.subgraph_mask = get_cut_graph_mask(cls.edge_index, n1)

        egnn_config.update({"reflect_equiv": False})
        cls.egnn_no_reflect_equiv = EGNN(**egnn_config)
        left_config.update({"reflect_equiv": False})
        cls.leftnet_no_reflect_equiv = LEFTNet(**left_config)

        rot_x = torch.tensor(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ],
            dtype=default_float,
        )
        rot_y = torch.tensor(
            [
                [np.cos(alpha), 0, np.sin(alpha)],
                [0, 1, 0],
                [-np.sin(alpha), 0, np.cos(alpha)],
            ],
            dtype=default_float,
        )
        cls.rot = torch.matmul(rot_y, rot_x).double()
        cls.trans = torch.rand(3) * 100

        cls.egnn_no_reflect_equiv.apply(init_weights)
        cls.leftnet.apply(init_weights)
        cls.leftnet_no_reflect_equiv.apply(init_weights)

        cls.models = [cls.egnn, cls.leftnet]
        cls.models_no_reflect_equiv = [
            cls.egnn_no_reflect_equiv,
            cls.leftnet_no_reflect_equiv,
        ]

    def test_rotation(self):
        pos_rot = torch.concat(
            [
                com(torch.matmul(self.pos[: self.n1], self.rot)),
                self.pos[self.n1 :],
            ],
            dim=0,
        ).double()
        for model in self.models:
            _h, _pos, _edge_attr = model.forward(
                self.h,
                self.pos,
                self.edge_index,
                self.edge_attr,
                subgraph_mask=self.subgraph_mask,
            )

            _h_rot, _pos_rot, _edge_attr_rot = model.forward(
                self.h,
                pos_rot,
                self.edge_index,
                self.edge_attr,
                subgraph_mask=self.subgraph_mask,
            )

            print(
                tensor_relative_diff(_h, _h_rot),
            )
            print(
                tensor_relative_diff(
                    torch.concat(
                        [torch.matmul(_pos[: self.n1], self.rot), _pos[self.n1 :]],
                        dim=0,
                    ).double(),
                    _pos_rot,
                )
            )
            self.assertTrue(tensor_relative_diff(_h, _h_rot) < EPS)
            if _edge_attr is not None:
                self.assertTrue(tensor_relative_diff(_edge_attr, _edge_attr_rot) < EPS)
            self.assertTrue(
                tensor_relative_diff(
                    torch.concat(
                        [torch.matmul(_pos[: self.n1], self.rot), _pos[self.n1 :]],
                        dim=0,
                    ).double(),
                    _pos_rot,
                )
                < EPS
            )

    def test_translation(self):
        for model in self.models:
            _h, _pos, _edge_attr = model.forward(
                self.h,
                self.pos,
                self.edge_index,
                self.edge_attr,
                subgraph_mask=self.subgraph_mask,
            )

            pos_trans = torch.concat(
                [com(self.pos[: self.n1] + self.trans), self.pos[self.n1 :]], dim=0
            )
            _h_trans, _pos_trans, _edge_attr_trans = model.forward(
                self.h,
                pos_trans,
                self.edge_index,
                self.edge_attr,
                subgraph_mask=self.subgraph_mask,
            )

            print(
                tensor_relative_diff(_h, _h_trans),
            )
            print(
                tensor_relative_diff(
                    torch.concat([_pos[: self.n1], _pos[self.n1 :]], dim=0),
                    _pos_trans,
                )
            )
            self.assertTrue(tensor_relative_diff(_h, _h_trans) < EPS)
            if _edge_attr is not None:
                self.assertTrue(
                    tensor_relative_diff(_edge_attr, _edge_attr_trans) < EPS
                )
            self.assertTrue(
                tensor_relative_diff(
                    torch.concat([_pos[: self.n1], _pos[self.n1 :]], dim=0),
                    _pos_trans,
                )
                < EPS
            )

    def test_break_graph_completely(self):
        """Add subgraph_mask is *NOT* the same as completely breaking a graph."""
        for model in self.models:
            _h, _pos, _edge_attr = model.forward(
                self.h,
                self.pos,
                self.edge_index,
                self.edge_attr,
                subgraph_mask=self.subgraph_mask,
            )

            edge_index_cut = []
            edge_attr_cut = []
            __edge_attr = []
            for ii, ele in enumerate(torch.transpose(self.edge_index, 1, 0)):
                if self.subgraph_mask[ii] == 1:
                    edge_index_cut.append(ele[None, :])
                    if _edge_attr is not None:
                        edge_attr_cut.append(self.edge_attr[ii][None, :])
                        __edge_attr.append(_edge_attr[ii][None, :])
            edge_index_cut = torch.transpose(torch.concat(edge_index_cut, dim=0), 1, 0)
            if _edge_attr is not None:
                edge_attr_cut = torch.concat(edge_attr_cut, dim=0)
                __edge_attr = torch.concat(__edge_attr, dim=0)

            _h_cut, _pos_cut, _edge_attr_cut = model.forward(
                self.h, self.pos, edge_index_cut, edge_attr_cut
            )

            print(tensor_relative_diff(_h, _h_cut))
            print(tensor_relative_diff(_pos, _pos_cut))
            self.assertTrue(tensor_relative_diff(_h, _h_cut) > LARGE_EPS)
            self.assertTrue(
                tensor_relative_diff(
                    _pos,
                    _pos_cut,
                )
                > LARGE_EPS
            )

    def test_subgraph_reflection(self):
        pos = self.pos.detach().clone()
        pos[: self.n1, 2] = -pos[: self.n1, 2]
        for model in self.models_no_reflect_equiv:
            _h, _pos, _edge_attr = model.forward(
                self.h,
                self.pos,
                self.edge_index,
                self.edge_attr,
                subgraph_mask=self.subgraph_mask,
            )

            _h_r, _pos_r, _edge_attr_r = model.forward(
                self.h,
                pos,
                self.edge_index,
                self.edge_attr,
                subgraph_mask=self.subgraph_mask,
            )
            print(
                tensor_relative_diff(_h[self.n1 :], _h_r[self.n1 :]),
                tensor_relative_diff(_pos[self.n1 :], _pos_r[self.n1 :]),
            )
            self.assertTrue(
                tensor_relative_diff(_pos[self.n1 :], _pos_r[self.n1 :]) > 1e-7
            )

    def test_subgraph_position_update(self):
        """Change the geometry of one fragment should be seen by the others"""
        pos = self.pos.detach().clone()
        pos[: self.n1] = com(torch.rand_like(pos[: self.n1]) * 30)
        self.assertTrue(
            tensor_relative_diff(pos[self.n1 :], self.pos[self.n1 :]) < 1e-10
        )
        for model in self.models:
            _h, _pos, _edge_attr = model.forward(
                self.h,
                self.pos,
                self.edge_index,
                self.edge_attr,
                subgraph_mask=self.subgraph_mask,
            )

            _h_new, _pos_new, _edge_attr_new = model.forward(
                self.h,
                pos,
                self.edge_index,
                self.edge_attr,
                subgraph_mask=self.subgraph_mask,
            )
            print("_h: ", _h)
            print("_h_new: ", _h_new)
            print(
                tensor_relative_diff(_h[self.n1 :], _h_new[self.n1 :]),
                tensor_relative_diff(_pos[self.n1 :], _pos_new[self.n1 :]),
            )
            self.assertTrue(
                tensor_relative_diff(_h[self.n1 :], _h_new[self.n1 :]) > LARGE_EPS
            )
            self.assertTrue(
                tensor_relative_diff(_pos[self.n1 :], _pos_new[self.n1 :]) > LARGE_EPS
            )

    def test_separate_graphs_wo_edges(self):
        """If no e_ij between two parts, updating one would not change the other."""
        n1, n2 = 3, 2
        ntot = n1 + n2
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 0, 2, 3, 4], [1, 0, 2, 1, 2, 0, 4, 3]], dtype=torch.long
        )
        h = torch.rand(ntot, egnn_config["in_node_nf"])
        pos = torch.cat(
            [
                com(torch.rand(n1, 3)),
                com(torch.rand(n2, 3)),
            ],
            dim=0,
        )
        trans = torch.rand(3)
        edge_attr = torch.rand(edge_index.size(1), egnn_config["in_edge_nf"])

        for model in self.models:
            _h, _pos, _edge_attr = model.forward(
                h,
                pos,
                edge_index,
                edge_attr,
                subgraph_mask=None,
            )

            pos_trans = torch.concat([com(pos[:n1] + trans), pos[n1:]], dim=0)
            _h_trans, _pos_trans, _edge_attr_trans = model.forward(
                h,
                pos_trans,
                edge_index,
                edge_attr,
                subgraph_mask=None,
            )

            self.assertTrue(tensor_relative_diff(_h, _h_trans) < EPS)
            self.assertTrue(
                tensor_relative_diff(
                    torch.concat([_pos[:n1], _pos[n1:]], dim=0),
                    _pos_trans,
                )
                < EPS
            )

            pos[:n1] = com(torch.rand_like(pos[:n1]))
            _h_new, _pos_new, _edge_attr_trans = model.forward(
                h,
                pos,
                edge_index,
                edge_attr,
                subgraph_mask=None,
            )
            self.assertTrue(tensor_relative_diff(_h[n1:], _h_new[n1:]) < EPS)
            self.assertTrue(tensor_relative_diff(_pos[n1:], _pos_new[n1:]) < EPS)
