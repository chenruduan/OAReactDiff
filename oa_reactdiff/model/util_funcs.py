"""Utility functions for model"""
import torch
from torch import Tensor


def move_by_com(pos):
    return pos - torch.mean(pos, dim=0)


def coord2cross(x, edge_index, norm_constant=1):
    row, col = edge_index
    cross = torch.cross(x[row], x[col], dim=1)
    norm = torch.linalg.norm(cross, dim=1, keepdim=True)
    cross = cross / (norm + norm_constant)
    return cross


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method: str
):
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor

    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


def get_ji_bond_index(bond_atom_indices: Tensor) -> Tensor:
    r"""Get the index for e_ji
    for example, bond_atom_indices = [[0, 1], [1, 0]], returns [1, 0]

    Args:
        bond_atom_indices (Tensor): (2, n_bonds) for ij

    Returns:
        Tensor: index for ji
    """
    bond_atom_indices = torch.transpose(bond_atom_indices, 0, 1)
    _index = torch.LongTensor([1, 0])
    reverse_bond_atom_indices = bond_atom_indices[:, _index]
    bond_ji_index = []
    for ij in range(bond_atom_indices.shape[0]):
        bond_ji_index.append(
            torch.where(
                (bond_atom_indices == reverse_bond_atom_indices[ij]).all(dim=1)
            )[0]
        )
    return torch.concat(bond_ji_index).long()


def symmetrize_edge(edge_attr: Tensor, edge_ji_indices: Tensor) -> Tensor:
    return (edge_attr + edge_attr[edge_ji_indices]) / 2
