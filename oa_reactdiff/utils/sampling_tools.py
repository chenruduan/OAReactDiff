from typing import List
import torch
import numpy as np
from oa_reactdiff.utils import bond_analyze


def write_xyz(mol, dataset_info, xyzfile="tmp.xyz"):
    atom_decoder = dataset_info["atom_decoder"]
    n_atom = mol["atom"].size(0)
    with open(xyzfile, "w") as fo:
        fo.write(str(n_atom) + "\n\n")
        for ii in range(n_atom):
            pos = mol["pos"][ii].cpu().numpy()
            ele = atom_decoder[mol["atom"][ii]]
            _x = " ".join([str(__x) for __x in pos])
            fo.write(f"{ele} {_x}\n")


def check_stability(mol, dataset_info, debug=False):
    positions, atom_type = mol["pos"], mol["atom"]
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info["atom_decoder"]
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype="int")

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if dataset_info["name"] == "qm9":
                order = bond_analyze.get_bond_order(atom1, atom2, dist)
            else:
                raise KeyError("only qm9 is allowed!")
            # if i == 3 or j == 3:
            #     print(i, j, dist, order)
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[atom_decoder[atom_type_i]]
        # print(atom_decoder[atom_type_i], nr_bonds_i)
        if type(possible_bonds) == int:
            is_stable = possible_bonds >= nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print(
                "Invalid bonds for molecule %s with %d bonds"
                % (atom_decoder[atom_type_i], nr_bonds_i)
            )
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return int(molecule_stable), nr_stable_bonds, len(x)


def assemble_sample_inputs(
    atoms: List,
    device: torch.device = torch.device("cuda"),
    n_samples: int = 1,
    frag_type: bool = False,
):
    empty_site = torch.tensor([[1, 0, 0, 0, 0, 1]], device=device)
    if not frag_type:
        decoders = [
            {
                "H": [1, 0, 0, 0, 0, 1],
                "C": [0, 1, 0, 0, 0, 6],
                "N": [0, 0, 1, 0, 0, 7],
                "O": [0, 0, 0, 1, 0, 8],
                "F": [0, 0, 0, 0, 1, 9],
            }
        ] * 2
    else:
        decoders = [
            {
                "H": [1, 0, 0, 0, 0, 1, 0],
                "C": [0, 1, 0, 0, 0, 6, 0],
                "N": [0, 0, 1, 0, 0, 7, 0],
                "O": [0, 0, 0, 1, 0, 8, 0],
                "F": [0, 0, 0, 0, 1, 9, 0],
            },
            {
                "H": [1, 0, 0, 0, 0, 1, 1],
                "C": [0, 1, 0, 0, 0, 6, 1],
                "N": [0, 0, 1, 0, 0, 7, 1],
                "O": [0, 0, 0, 1, 0, 8, 1],
                "F": [0, 0, 0, 0, 1, 9, 1],
            },
        ]

    h0 = [
        torch.cat(
            [
                torch.tensor([decoders[ii % 2][atom] for atom in atoms], device=device)
                for _ in range(n_samples)
            ]
        )
        for ii in range(3)
    ]
    return h0


def write_single_xyz(xyzfile, natoms, out):
    C2A = {
        1: "H",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
    }
    with open(xyzfile, "w") as fo:
        fo.write(str(natoms) + "\n\n")
        for ele in out:
            ele = ele[: 3 + 5 + 1]
            x = ele[:3].cpu().numpy()
            _a = C2A[ele[-1].long().item()]
            _x = " ".join([str(__x) for __x in x])
            fo.write(f"{_a} {_x}\n")


def write_tmp_xyz(
    fragments_nodes, out_samples, idx=[0], prefix="gen", localpath="tmp", ex_ind=0
):
    TYPEMAP = {
        0: "react",
        1: "ts",
        2: "prod",
    }
    for ii in idx:
        st = TYPEMAP[ii]
        start_ind, end_ind = 0, 0
        for jj, natoms in enumerate(fragments_nodes[0]):
            _jj = jj + ex_ind
            xyzfile = f"{localpath}/{prefix}_{_jj}_{st}.xyz"
            end_ind += natoms.item()
            write_single_xyz(
                xyzfile,
                natoms.item(),
                out=out_samples[ii][start_ind:end_ind],
            )
            start_ind = end_ind
