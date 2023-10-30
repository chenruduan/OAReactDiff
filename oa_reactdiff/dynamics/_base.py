"""Base class for assembling fragments and performing model updates."""
from typing import Dict, List, Optional
import torch
from torch import nn

from oa_reactdiff.model import MLP, EGNN


class BaseDynamics(nn.Module):
    def __init__(
        self,
        model_config: Dict,
        fragment_names: List[str],
        node_nfs: List[int],
        edge_nf: int,
        condition_nf: int = 0,
        pos_dim: int = 3,
        update_pocket_coords: bool = True,
        condition_time: bool = True,
        edge_cutoff: Optional[float] = None,
        model: nn.Module = EGNN,
        device: torch.device = torch.device("cuda"),
        enforce_same_encoding: Optional[List] = None,
        source: Optional[Dict] = None,
    ) -> None:
        r"""Base dynamics class set up for denoising process.

        Args:
            model_config (Dict): config for the equivariant model.
            fragment_names (List[str]): list of names for fragments
            node_nfs (List[int]): list of number of input node attributues.
            edge_nf (int): number of input edge attributes.
            condition_nf (int): number of attributes for conditional generation.
            Defaults to 0.
            pos_dim (int): dimension for position vector. Defaults to 3.
            update_pocket_coords (bool): whether to update positions of everything.
                Defaults to True.
            condition_time (bool): whether to condition on time. Defaults to True.
            edge_cutoff (Optional[float]): cutoff for building intra-fragment edges.
                Defaults to None.
            model (Optional[nn.Module]): Module for equivariant model. Defaults to None.
        """
        super().__init__()
        assert len(node_nfs) == len(fragment_names)
        for nf in node_nfs:
            assert nf > pos_dim
        if "act_fn" not in model_config:
            model_config["act_fn"] = "swish"
        if "in_node_nf" not in model_config:
            model_config["in_node_nf"] = model_config["in_hidden_channels"]
        self.model_config = model_config
        self.node_nfs = node_nfs
        self.edge_nf = edge_nf
        self.condition_nf = condition_nf
        self.fragment_names = fragment_names
        self.pos_dim = pos_dim
        self.update_pocket_coords = update_pocket_coords
        self.condition_time = condition_time
        self.edge_cutoff = edge_cutoff
        self.device = device

        if model is None:
            model = EGNN
        self.model = model(**model_config)
        if source is not None:
            self.model.load_state_dict(source["model"])
        self.dist_dim = self.model.dist_dim if hasattr(self.model, "dist_dim") else 0

        self.embed_dim = model_config["in_node_nf"]
        self.edge_embed_dim = (
            model_config["in_edge_nf"] if "in_edge_nf" in model_config else 0
        )
        if condition_time:
            self.embed_dim -= 1
        if condition_nf > 0:
            self.embed_dim -= condition_nf
        assert self.embed_dim > 0

        self.build_encoders_decoders(enforce_same_encoding, source)
        del source

    def build_encoders_decoders(
        self,
        enfoce_name_encoding: Optional[List] = None,
        source: Optional[Dict] = None,
    ):
        r"""Build encoders and decoders for nodes and edges."""
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ii, name in enumerate(self.fragment_names):
            self.encoders.append(
                MLP(
                    in_dim=self.node_nfs[ii] - self.pos_dim,
                    out_dims=[2 * (self.node_nfs[ii] - self.pos_dim), self.embed_dim],
                    activation=self.model_config["act_fn"],
                    last_layer_no_activation=True,
                )
            )
            self.decoders.append(
                MLP(
                    in_dim=self.embed_dim,
                    out_dims=[
                        2 * (self.node_nfs[ii] - self.pos_dim),
                        self.node_nfs[ii] - self.pos_dim,
                    ],
                    activation=self.model_config["act_fn"],
                    last_layer_no_activation=True,
                )
            )
        if enfoce_name_encoding is not None:
            for ii in enfoce_name_encoding:
                self.encoders[ii] = self.encoders[0]
                self.decoders[ii] = self.decoders[0]
        if source is not None:
            self.encoders.load_state_dict(source["encoders"])
            self.decoders.load_state_dict(source["decoders"])

        if self.edge_embed_dim > 0:
            self.edge_encoder = MLP(
                in_dim=self.edge_nf,
                out_dims=[2 * self.edge_nf, self.edge_embed_dim],
                activation=self.model_config["act_fn"],
                last_layer_no_activation=True,
            )
            self.edge_decoder = MLP(
                in_dim=self.edge_embed_dim + self.dist_dim,
                out_dims=[2 * self.edge_nf, self.edge_nf],
                activation=self.model_config["act_fn"],
                last_layer_no_activation=True,
            )
        else:
            self.edge_encoder, self.edge_decoder = None, None

    def forward(self):
        raise NotImplementedError
