import unittest

import torch
from torch import Tensor, tensor

from oa_reactdiff.utils import (
    get_edges_index,
    get_subgraph_mask,
    get_n_frag_switch,
    get_mask_for_frag,
)


class TestBasics(unittest.TestCase):
    def test_get_mask_for_frag(self):
        natms = Tensor([2, 0, 3]).long()
        res = get_mask_for_frag(natms)
        self.assertTrue(torch.allclose(res, Tensor([0, 0, 2, 2, 2]).long()))

    def test_get_n_frag_switch(self):
        natm_list = [tensor([2, 0]), tensor([1, 3]), tensor([3, 2])]
        res = get_n_frag_switch(natm_list)
        self.assertTrue(torch.allclose(res, tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])))

    def test_get_subgraph_mask(self):
        edge_index = tensor(
            [
                [0, 0, 1, 1, 2, 2],
                [1, 2, 0, 2, 0, 1],
            ]
        )
        n_frag_switch = tensor([0, 0, 1])
        res = get_subgraph_mask(edge_index, n_frag_switch)
        self.assertTrue(torch.allclose(res, tensor([1, 0, 1, 0, 0, 0])))

    def test_complete_generation(self):
        natm_inorg_node = torch.tensor([2, 0])
        natm_org_edge = torch.tensor([2, 3])
        natm_org_node = torch.tensor([1, 2])

        inorg_node_mask = get_mask_for_frag(natm_inorg_node)
        org_edge_mask = get_mask_for_frag(natm_org_edge)
        org_node_mask = get_mask_for_frag(natm_org_node)

        n_frag_switch = get_n_frag_switch(
            [natm_inorg_node, natm_org_edge, natm_org_node]
        )
        self.assertTrue(
            torch.allclose(n_frag_switch, tensor([0, 0, 1, 1, 1, 1, 1, 2, 2, 2]))
        )

        combined_mask = torch.cat((inorg_node_mask, org_edge_mask, org_node_mask))
        _edge_index = get_edges_index(combined_mask)
        self.assertTrue(
            _edge_index.shape == (2, 2 * 5 + 0 * 5 + 2 * 5 + 3 * 5 + 1 * 5 + 2 * 5)
        )
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        self.assertTrue(
            edge_index.shape == (2, 2 * 5 + 0 * 5 + 2 * 5 + 3 * 5 + 1 * 5 + 2 * 5 - 10)
        )

        subgraph_mask = get_subgraph_mask(edge_index, n_frag_switch)
        self.assertTrue(torch.sum(subgraph_mask) == 2 + 2 + 6 + 2)
