"""Test model forward pass and equivariance."""
import unittest
import torch
from pathlib import Path

from oa_reactdiff.dataset.transition1x import ProcessedTS1x


def get_project_root() -> Path:
    return Path(__file__).parent.parent


class TestTS1x(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEST_ROOT = get_project_root()
        cls.dataset = ProcessedTS1x(
            npz_path=f"{TEST_ROOT}/../data/transition1x/train.pkl",
            center=True,
            pad_fragments=0,
            device="cpu",
            zero_charge=False,
            remove_h=False,
            single_frag_only=False,
            swapping_react_prod=False,
            use_by_ind=True,
        )

    def test_len(self):
        assert len(self.dataset) == 9000

    def test_one(self):
        data = self.dataset[42]
        assert isinstance(data, dict)

        unique_elements = []
        n_frag = 0
        for k, v in data.items():
            assert torch.is_tensor(v)
            kk = k.split("_")
            if kk[-1].isdigit():
                n_frag = max(n_frag, int(kk[-1]))
            if kk[0] not in unique_elements:
                unique_elements.append(kk[0])
        assert set(unique_elements) == set(
            ["size", "pos", "one", "charge", "mask", "condition"]
        )
        assert n_frag + 1 == 3
