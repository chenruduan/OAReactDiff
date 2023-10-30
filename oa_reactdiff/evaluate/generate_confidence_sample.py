from typing import List
import time
import os
import numpy as np
import torch
import pickle
import argparse
from uuid import uuid4

from torch.utils.data import DataLoader

from oa_reactdiff.trainer.pl_trainer import DDPMModule
from oa_reactdiff.dataset.transition1x import ProcessedTS1x
from oa_reactdiff.analyze.rmsd import batch_rmsd
from oa_reactdiff.evaluate.utils import (
    set_new_schedule,
    inplaint_batch,
    samples_to_pos_charge,
)

EV2KCALMOL = 23.06
AU2KCALMOL = 627.5


def assemble_filename(config):
    _id = str(uuid4()).split("-")[0]
    filename = f"conf-uuid-{_id}-"
    for k, v in config.items():
        filename += f"{k}-{v}_"
    filename += ".pkl"
    print(filename)
    return filename


parser = argparse.ArgumentParser(description="get training params")
parser.add_argument("--bz", dest="bz", default=32, type=int, help="batch size")
parser.add_argument(
    "--timesteps", dest="timesteps", default=150, type=int, help="timesteps"
)
parser.add_argument(
    "--resamplings", dest="resamplings", default=2, type=int, help="resamplings"
)
parser.add_argument(
    "--jump_length", dest="jump_length", default=2, type=int, help="jump_length"
)
parser.add_argument("--repeats", dest="repeats", default=1, type=int, help="repeats")
parser.add_argument(
    "--partition", dest="partition", default="valid", type=str, help="partition"
)
parser.add_argument(
    "--dataset", dest="dataset", default="transition1x", type=str, help="dataset"
)
parser.add_argument(
    "--single_frag_only",
    dest="single_frag_only",
    default=0,
    type=int,
    help="single_frag_only",
)
parser.add_argument(
    "--model", dest="model", default="leftnet_2074", type=str, help="model"
)
parser.add_argument("--power", dest="power", default="2", type=str, help="power")
parser.add_argument(
    "--position_key",
    dest="position_key",
    default="positions",
    type=str,
    help="position_key",
)

args = parser.parse_args()
print("args: ", args)

config = dict(
    model=args.model,
    dataset=args.dataset,
    partition=args.partition,
    timesteps=args.timesteps,
    bz=args.bz,
    resamplings=args.resamplings,
    jump_length=args.jump_length,
    repeats=args.repeats,
    max_batch=-1,
    shuffle=True,
    single_frag_only=args.single_frag_only,
    noise_schedule="polynomial_" + args.power,
    position_key=args.position_key,
)

print("loading ddpm trainer...")
device = torch.device("cuda")
tspath = "/home/ubuntu/efs/TSDiffusion/oa_reactdiff/trainer/ckpt/TSDiffusion-TS1x-All"
checkpoints = {
    "leftnet_2074": f"{tspath}/leftnet-8-70b75beeaac1/ddpm-epoch=2074-val-totloss=531.18.ckpt",
    "egnn": f"{tspath}/egnn-1-7d0e388fa0fd/ddpm-epoch=759-val-totloss=616.42.ckpt",
    "leftnet_wo_oa": f"{tspath}/leftnet-10-da396de30744_wo_oa/ddpm-epoch=149-val-totloss=600.87.ckpt",
    "leftnet_wo_oa_aligned": f"{tspath}/leftnet-10-d13a2c2bace6_wo_oa_align/ddpm-epoch=779-val-totloss=747.10.ckpt",
    "leftnet_wo_oa_aligned_early": f"{tspath}/leftnet-10-d13a2c2bace6_wo_oa_align/ddpm-epoch=719-val-totloss=680.64.ckpt",
}
ddpm_trainer = DDPMModule.load_from_checkpoint(
    checkpoint_path=checkpoints[config["model"]],
    map_location=device,
)
ddpm_trainer = set_new_schedule(
    ddpm_trainer, timesteps=config["timesteps"], noise_schedule=config["noise_schedule"]
)

print("loading dataset...")
dataset = ProcessedTS1x(
    npz_path=f"../data/{args.dataset}/{args.partition}.pkl",
    center=True,
    pad_fragments=0,
    device="cuda",
    zero_charge=False,
    remove_h=False,
    single_frag_only=config["single_frag_only"],
    swapping_react_prod=False,
    use_by_ind=True,
    position_key=config["position_key"],
)
print("# of points:", len(dataset))
loader = DataLoader(
    dataset,
    batch_size=config["bz"],
    shuffle=config["shuffle"],
    num_workers=0,
    collate_fn=dataset.collate_fn,
)

print("evaluating...")
speices = ["reactant", "transition_state", "product"]
keys = ["num_atoms", "charges", "position"]

for num_repeat in range(config["repeats"]):
    print("num_repeat: ", num_repeat)
    _rmsds, _genEs = [], []
    filename = assemble_filename(config)

    data = {}
    for s in speices:
        data[s] = {}
        for k in keys:
            data[s][k] = []
    for s in ["target", "rmsd"]:
        data[s] = []

    for ii, batch in enumerate(loader):
        print("batch_idx: ", ii)
        time_start = time.time()
        if ii == config["max_batch"]:
            break

        # TS gen
        out_samples, xh_fixed, fragments_nodes = inplaint_batch(
            batch,
            ddpm_trainer,
            resamplings=config["resamplings"],
            jump_length=config["jump_length"],
            frag_fixed=[0, 2],
        )
        pos, z, natoms = samples_to_pos_charge(out_samples, fragments_nodes)
        _rmsds = batch_rmsd(
            fragments_nodes,
            out_samples,
            xh_fixed,
            idx=1,
            threshold=0.5,
        )
        for s in speices:
            data[s]["position"] += pos[s]
            data[s]["charges"] += z
            data[s]["num_atoms"] += natoms
        data["rmsd"] += _rmsds
        data["target"] += [1 if _r < 0.2 else 0 for _r in _rmsds]
        print("time cost: ", time.time() - time_start)
        print(
            "rmsds: ",
            [round(_x, 2) for _x in data["rmsd"]],
            np.mean(data["rmsd"]),
            np.median(data["rmsd"]),
        )

        # # R gen
        # out_samples, xh_fixed, fragments_nodes = inplaint_batch(
        #     batch,
        #     ddpm_trainer,
        #     resamplings=config["resamplings"] * 2,
        #     jump_length=config["jump_length"] * 2,
        #     frag_fixed=[1, 2]
        # )
        # pos, z, natoms = samples_to_pos_charge(out_samples, fragments_nodes)
        # _rmsds = batch_rmsd(
        #     fragments_nodes,
        #     out_samples,
        #     xh_fixed,
        #     idx=0,
        #     threshold=0.5,
        # )
        # for s in speices:
        #     data[s]["position"] += pos[s]
        #     data[s]["charges"] += z
        #     data[s]["num_atoms"] += natoms
        # data["rmsd"] += _rmsds
        # data["target"] += [1 if _r < 0.2 else 0 for _r in _rmsds]
        # print("time cost: ", time.time() - time_start)
        # print("rmsds: ", [round(_x, 2) for _x in _rmsds], np.mean(_rmsds), np.std(_rmsds))

        with open(f"samples/{filename}", "wb") as fo:
            pickle.dump(data, fo)
