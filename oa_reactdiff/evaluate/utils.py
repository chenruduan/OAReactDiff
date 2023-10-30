from typing import List
import torch

from oa_reactdiff.trainer.pl_trainer import DDPMModule
from oa_reactdiff.diffusion._schedule import DiffSchedule, PredefinedNoiseSchedule
from oa_reactdiff.diffusion._normalizer import FEATURE_MAPPING
from oa_reactdiff.analyze.geomopt import calc_deltaE, compute_efh

EV2KCALMOL = 23.06
AU2KCALMOL = 627.5
device = torch.device("cuda")


def set_new_schedule(
    ddpm_trainer: DDPMModule,
    timesteps: int = 250,
    device: torch.device = torch.device("cuda"),
    noise_schedule: str = "polynomial_2",
) -> DDPMModule:
    precision: float = 1e-5

    gamma_module = PredefinedNoiseSchedule(
        noise_schedule=noise_schedule,
        timesteps=timesteps,
        precision=precision,
    )
    schedule = DiffSchedule(
        gamma_module=gamma_module, norm_values=ddpm_trainer.ddpm.norm_values
    )
    ddpm_trainer.ddpm.schedule = schedule
    ddpm_trainer.ddpm.T = timesteps
    return ddpm_trainer.to(device)


def inplaint_batch(
    batch: List,
    ddpm_trainer: DDPMModule,
    resamplings: int = 1,
    jump_length: int = 1,
    frag_fixed: List = [0, 2],
):
    representations, conditions = batch
    xh_fixed = [
        torch.cat(
            [repre[feature_type] for feature_type in FEATURE_MAPPING],
            dim=1,
        )
        for repre in representations
    ]
    n_samples = representations[0]["size"].size(0)
    fragments_nodes = [repre["size"] for repre in representations]
    out_samples, _ = ddpm_trainer.ddpm.inpaint(
        n_samples=n_samples,
        fragments_nodes=fragments_nodes,
        conditions=conditions,
        return_frames=1,
        resamplings=resamplings,
        jump_length=jump_length,
        timesteps=None,
        xh_fixed=xh_fixed,
        frag_fixed=frag_fixed,
    )
    return out_samples[0], xh_fixed, fragments_nodes


def batch_ts_deltaE(bz, xc="wb97x", localpath="tmp"):
    deltaEs = []
    for ii in range(bz):
        deltaEs.append(
            calc_deltaE(
                f"{localpath}/sample_{ii}_ts.xyz",
                f"{localpath}/gen_{ii}_ts.xyz",
                xc=xc,
            )
            * EV2KCALMOL
        )
        print("----")
    return deltaEs


def batch_E(bz, prefix="gen"):
    Es = []
    for ii in range(bz):
        mf, _, _ = compute_efh(
            f"tmp/{prefix}_{ii}_ts.xyz", f=False, hess=False, return_metrics=False
        )
        Es.append(mf.e_tot * AU2KCALMOL)
    return Es


def samples_to_pos_charge(out_samples, fragments_nodes):
    x_r = torch.tensor_split(
        out_samples[0], torch.cumsum(fragments_nodes[0], dim=0).to("cpu")[:-1]
    )
    x_ts = torch.tensor_split(
        out_samples[1], torch.cumsum(fragments_nodes[0], dim=0).to("cpu")[:-1]
    )
    x_p = torch.tensor_split(
        out_samples[2], torch.cumsum(fragments_nodes[0], dim=0).to("cpu")[:-1]
    )
    pos = {
        "reactant": [_x[:, :3].cpu().numpy() for _x in x_r],
        "transition_state": [_x[:, :3].cpu().numpy() for _x in x_ts],
        "product": [_x[:, :3].cpu().numpy() for _x in x_p],
    }
    z = [_x[:, -1].long().cpu().numpy() for _x in x_r]
    natoms = [f.cpu().item() for f in fragments_nodes[0]]
    return pos, z, natoms
