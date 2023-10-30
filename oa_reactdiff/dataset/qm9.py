import numpy as np
import torch

from oa_reactdiff.dataset.base_dataset import BaseDataset, ATOM_MAPPING

n_element = len(list(ATOM_MAPPING.keys()))


class BaseQM9(BaseDataset):
    def __init__(
        self,
        npz_path,
        center=True,
        zero_charge=False,
        device="cpu",
        remove_h=False,
    ) -> None:
        super().__init__(
            npz_path=npz_path,
            center=center,
            zero_charge=zero_charge,
            device=device,
            remove_h=remove_h,
        )
        if self.remove_h:
            pos = self.raw_dataset["positions"]
            charges = self.raw_dataset["charges"]
            num_atoms = self.raw_dataset["num_atoms"]

            mask = self.raw_dataset["charges"] > 1
            new_positions = np.zeros_like(pos)
            new_charges = np.zeros_like(charges)
            for i in range(new_positions.shape[0]):
                m = mask[i]
                p = pos[i][m]  # positions to keep
                c = charges[i][m]  # Charges to keep
                n = np.sum(m)
                new_positions[i, :n, :] = p
                new_charges[i, :n] = c

            self.raw_dataset["positions"] = new_positions
            self.raw_dataset["charges"] = new_charges
            self.raw_dataset["num_atoms"] = np.sum(
                self.raw_dataset["charges"] > 0, axis=1
            )

        self.n_samples = len(self.raw_dataset["charges"])
        self.data = {}

    def get_subsets(self):
        hasN, hasO, hasF = [], [], []
        for ii in range(self.n_samples):
            charges = self.raw_dataset["charges"][ii]
            unique_charges = np.unique(charges)
            if set(unique_charges) <= set([0, 1, 6, 8]) and 8 in set(unique_charges):
                hasO.append(ii)
            if set(unique_charges) <= set([0, 1, 6, 7]) and 7 in set(unique_charges):
                hasN.append(ii)
            if set(unique_charges) <= set([0, 1, 6, 9]) and 9 in set(unique_charges):
                hasF.append(ii)
        self.hasO_set = {key: val[hasO] for key, val in self.raw_dataset.items()}
        self.hasN_set = {key: val[hasN] for key, val in self.raw_dataset.items()}
        self.hasF_set = {key: val[hasF] for key, val in self.raw_dataset.items()}


class ProcessedQM9(BaseQM9):
    def __init__(
        self,
        npz_path,
        center=True,
        pad_fragments=2,
        device="cpu",
        zero_charge=False,
        remove_h=False,
        **kwargs,
    ):
        super().__init__(
            npz_path=npz_path,
            center=center,
            device=device,
            zero_charge=zero_charge,
            remove_h=remove_h,
        )

        self.n_fragments = pad_fragments + 1
        self.device = torch.device(device)

        n_samples = len(self.raw_dataset["charges"])
        self.n_samples = n_samples

        self.data = {}
        self.process_molecules("raw_dataset", n_samples, idx=0)

        for idx in range(pad_fragments):
            self.patch_dummy_molecules(idx + 1)

        self.data["condition"] = [
            torch.zeros(
                size=(1, 1),
                dtype=torch.int64,
                device=self.device,
            )
            for _ in range(self.n_samples)
        ]


class ProcessedDoubleQM9(BaseQM9):
    def __init__(
        self,
        npz_path,
        center=True,
        pad_fragments=1,
        device="cpu",
        zero_charge=False,
        remove_h=False,
        **kwargs,
    ):
        super().__init__(
            npz_path=npz_path,
            center=center,
            device=device,
            zero_charge=zero_charge,
            remove_h=remove_h,
        )

        self.n_fragments = pad_fragments + 2
        self.device = torch.device(device)
        n_samples = len(self.raw_dataset["charges"])
        self.n_samples = len(self.raw_dataset["charges"])

        self.get_subsets()
        self.get_pairs()

        self.data = {}
        self.process_molecules("frag1_data", n_samples, idx=0)
        self.process_molecules("frag2_data", n_samples, idx=1)

        for idx in range(pad_fragments):
            self.patch_dummy_molecules(idx + 2)

        self.data["condition"] = [
            torch.zeros(
                size=(1, 1),
                dtype=torch.int64,
                device=self.device,
            )
            for _ in range(self.n_samples)
        ]

    def get_pairs(self):
        self.frag1_data, self.frag2_data = {}, {}
        frag1_O_idx_1sthalf = np.random.choice(
            len(self.hasO_set["charges"]),
            int(self.n_samples / 2),
            replace=True,
        )
        frag2_N_idx_1sthalf = np.random.choice(
            len(self.hasN_set["charges"]),
            int(self.n_samples / 2),
            replace=True,
        )
        frag1_N_idx_2ndhalf = np.random.choice(
            len(self.hasN_set["charges"]),
            int(self.n_samples / 2),
            replace=True,
        )
        frag2_O_idx_2ndhalf = np.random.choice(
            len(self.hasO_set["charges"]),
            int(self.n_samples / 2),
            replace=True,
        )
        self.frag1_data = {
            key: np.concatenate(
                [
                    self.hasO_set[key][frag1_O_idx_1sthalf],
                    self.hasN_set[key][frag1_N_idx_2ndhalf],
                ],
                axis=0,
            )
            for key in self.raw_dataset
        }
        self.frag2_data = {
            key: np.concatenate(
                [
                    self.hasN_set[key][frag2_N_idx_1sthalf],
                    self.hasO_set[key][frag2_O_idx_2ndhalf],
                ],
                axis=0,
            )
            for key in self.raw_dataset
        }


class ProcessedTripleQM9(BaseQM9):
    def __init__(
        self,
        npz_path,
        center=True,
        pad_fragments=0,
        device="cpu",
        zero_charge=False,
        remove_h=False,
        **kwargs,
    ):
        super().__init__(
            npz_path=npz_path,
            center=center,
            device=device,
            zero_charge=zero_charge,
            remove_h=remove_h,
        )

        self.n_fragments = pad_fragments + 3
        self.device = torch.device(device)
        n_samples = len(self.raw_dataset["charges"])
        self.n_samples = len(self.raw_dataset["charges"])

        self.get_subsets()
        self.get_pairs()

        self.data = {}
        self.process_molecules("frag1_data", n_samples, idx=0)
        self.process_molecules("frag2_data", n_samples, idx=1)
        self.process_molecules("frag3_data", n_samples, idx=2)

        for idx in range(pad_fragments):
            self.patch_dummy_molecules(idx + 3)

        self.data["condition"] = [
            torch.zeros(
                size=(1, 1),
                dtype=torch.int64,
                device=self.device,
            )
            for _ in range(self.n_samples)
        ]

    def get_pairs(self):
        n1 = int(self.n_samples / 3)
        n2 = int(self.n_samples / 3)
        n3 = self.n_samples - n1 - n2
        self.frag1_data, self.frag2_data = {}, {}
        frag1_O_idx_1_3 = np.random.choice(
            len(self.hasO_set["charges"]),
            n1,
            replace=True,
        )
        frag2_N_idx_1_3 = np.random.choice(
            len(self.hasN_set["charges"]),
            n1,
            replace=True,
        )
        frag3_F_idx_1_3 = np.random.choice(
            len(self.hasF_set["charges"]),
            n1,
            replace=True,
        )
        frag1_F_idx_2_3 = np.random.choice(
            len(self.hasF_set["charges"]),
            n2,
            replace=True,
        )
        frag2_O_idx_2_3 = np.random.choice(
            len(self.hasO_set["charges"]),
            n2,
            replace=True,
        )
        frag3_N_idx_2_3 = np.random.choice(
            len(self.hasN_set["charges"]),
            n2,
            replace=True,
        )
        frag1_N_idx_3_3 = np.random.choice(
            len(self.hasN_set["charges"]),
            n3,
            replace=True,
        )
        frag2_F_idx_3_3 = np.random.choice(
            len(self.hasF_set["charges"]),
            n3,
            replace=True,
        )
        frag3_O_idx_3_3 = np.random.choice(
            len(self.hasO_set["charges"]),
            n3,
            replace=True,
        )
        self.frag1_data = {
            key: np.concatenate(
                [
                    self.hasO_set[key][frag1_O_idx_1_3],
                    self.hasF_set[key][frag1_F_idx_2_3],
                    self.hasN_set[key][frag1_N_idx_3_3],
                ],
                axis=0,
            )
            for key in self.raw_dataset
        }
        self.frag2_data = {
            key: np.concatenate(
                [
                    self.hasN_set[key][frag2_N_idx_1_3],
                    self.hasO_set[key][frag2_O_idx_2_3],
                    self.hasF_set[key][frag2_F_idx_3_3],
                ],
                axis=0,
            )
            for key in self.raw_dataset
        }
        self.frag3_data = {
            key: np.concatenate(
                [
                    self.hasF_set[key][frag3_F_idx_1_3],
                    self.hasN_set[key][frag3_N_idx_2_3],
                    self.hasO_set[key][frag3_O_idx_3_3],
                ],
                axis=0,
            )
            for key in self.raw_dataset
        }
