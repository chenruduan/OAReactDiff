import torch

from pytorch_lightning import Trainer
from oa_reactdiff.diffusion._node_dist import SingleDistributionNodes
from oa_reactdiff.utils import bond_analyze


@torch.no_grad()
def sample_qm9(
    ddpm_trainer: Trainer,
    nodes_dist: SingleDistributionNodes,
    bz: int,
    n_samples: int,
    n_real: int = 1,
    n_fake: int = 2,
    device: torch.device = torch.device("cuda"),
):
    n_batch = int(n_samples / bz)
    mols = []
    pos_dim = ddpm_trainer.ddpm.pos_dim
    for _ in range(n_batch):
        fragments_nodes = [nodes_dist.sample(bz).to(device) for _ in range(n_real)]
        fragments_nodes += [torch.ones(bz, device=device).long() for _ in range(n_fake)]
        conditions = torch.zeros((bz, 1), device=device)

        out_samples, out_masks = ddpm_trainer.ddpm.sample(
            n_samples=bz,
            fragments_nodes=fragments_nodes,
            conditions=conditions,
            return_frames=1,
            timesteps=None,
        )
        sample_idxs = torch.cat(
            [torch.tensor([0], device=device), torch.cumsum(fragments_nodes[0], dim=0)]
        )
        for ii in range(bz):
            _start, _end = sample_idxs[ii], sample_idxs[ii + 1]
            mols.append(
                {
                    "pos": out_samples[0][0][_start:_end, :pos_dim].detach().cpu(),
                    "atom": torch.argmax(
                        out_samples[0][0][_start:_end, pos_dim:-1].detach().cpu(),
                        dim=1,
                    ),
                }
            )
    return mols
