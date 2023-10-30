from typing import List
import time
import os
import numpy as np
import torch
import pickle
import argparse

from torch.utils.data import DataLoader

from oa_reactdiff.trainer.pl_trainer import DDPMModule
from oa_reactdiff.dataset.transition1x import ProcessedTS1x
from oa_reactdiff.analyze.rmsd import batch_rmsd
from oa_reactdiff.evaluate.utils import (
    set_new_schedule,
    inplaint_batch,
)

EV2KCALMOL = 23.06
AU2KCALMOL = 627.5


parser = argparse.ArgumentParser(description="get training params")
parser.add_argument("--bz", dest="bz", default=64, type=int, help="batch size")
parser.add_argument(
    "--timesteps", dest="timesteps", default=250, type=int, help="timesteps"
)
parser.add_argument(
    "--resamplings", dest="resamplings", default=5, type=int, help="resamplings"
)
parser.add_argument(
    "--jump_length", dest="jump_length", default=5, type=int, help="jump_length"
)
parser.add_argument("--repeats", dest="repeats", default=5, type=int, help="repeats")
parser.add_argument(
    "--partition", dest="partition", default="valid", type=str, help="partition"
)
parser.add_argument(
    "--single_frag_only",
    dest="single_frag_only",
    default=1,
    type=int,
    help="single_frag_only",
)
parser.add_argument("--model", dest="model", default="leftnet", type=str, help="model")
parser.add_argument("--power", dest="power", default="2", type=str, help="power")

args = parser.parse_args()
print("args: ", args)

config = dict(
    model=args.model,
    partition=args.partition,
    timesteps=args.timesteps,
    bz=args.bz,
    resamplings=args.resamplings,
    jump_length=args.jump_length,
    repeats=args.repeats,
    max_batch=-1,
    shuffle=False,
    single_frag_only=args.single_frag_only,
    noise_schedule="polynomial_" + args.power,
)

filename = ""
for k, v in config.items():
    filename += f"{k}-{v}_"
filename += ".pkl"
print(filename)

print("loading ddpm trainer...")
device = torch.device("cuda")
tspath = "/home/ubuntu/efs/2Dto3D_ReactGen/oa_reactdiff/trainer/ckpt/TSDiffusion-TS1x"
checkpoints = {
    "chiral": f"{tspath}/5edcbc9baced/ddpm-epoch=4159-val-totloss=585.78.ckpt",
    "leftnet_legacy": f"{tspath}/leftnet-78c7590798bc/ddpm-epoch=1059-val-totloss=648.90.ckpt",
    "leftnet4": f"{tspath}/leftnet-4-48f308df7ec4/ddpm-epoch=809-val-totloss=587.27.ckpt",
    "leftnet_all": f"{tspath}-All/leftnet-4-77ae3fd23222/ddpm-epoch=1619-val-totloss=605.85.ckpt",
    # "leftnet_final": f"{tspath}-All/leftnet-8-17cf1d7b9324/ddpm-epoch=1289-val-totloss=536.86.ckpt",
    "leftnet_final": f"{tspath}-All/leftnet-8-70b75beeaac1/ddpm-epoch=1274-val-totloss=519.81.ckpt",
    "leftnet_1654": f"{tspath}-All/leftnet-8-70b75beeaac1/ddpm-epoch=1654-val-totloss=540.84.ckpt",
    "leftnet_1884": f"{tspath}-All/leftnet-8-70b75beeaac1/ddpm-epoch=1884-val-totloss=549.61.ckpt",
    "leftnet_2074": f"{tspath}-All/leftnet-8-70b75beeaac1/ddpm-epoch=2074-val-totloss=531.18.ckpt",
    "leftnet_2304": f"{tspath}-All/leftnet-8-70b75beeaac1/ddpm-epoch=2304-val-totloss=524.65.ckpt",
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
    npz_path=f"../data/transition1x/{args.partition}.pkl",
    center=True,
    pad_fragments=0,
    device="cuda",
    zero_charge=False,
    remove_h=False,
    single_frag_only=config["single_frag_only"],
    swapping_react_prod=False,
    use_by_ind=True,
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
if os.path.isfile(f"results/{filename}"):
    d = pickle.load(open(f"results/{filename}", "rb"))
    rmsds = d["rmsd"]
else:
    rmsds = []
TSEs = []
for num_repeat in range(config["repeats"]):
    print("num_repeat: ", num_repeat)
    _rmsds, _genEs = [], []
    for ii, batch in enumerate(loader):
        print("batch_idx: ", ii)
        time_start = time.time()
        if ii == config["max_batch"]:
            break
        out_samples, xh_fixed, fragments_nodes = inplaint_batch(
            batch,
            ddpm_trainer,
            resamplings=config["resamplings"],
            jump_length=config["jump_length"],
            frag_fixed=[0, 2],
        )
        # write_tmp_xyz(fragments_nodes, out_samples, idx=[0, 1, 2])
        # write_tmp_xyz(fragments_nodes, xh_fixed, idx=[1], prefix="sample")
        _rmsds += batch_rmsd(
            fragments_nodes,
            out_samples,
            xh_fixed,
            idx=1,
            threshold=0.5,
        )
        print("time cost: ", time.time() - time_start)
        print(
            "rmsds: ", [round(_x, 2) for _x in _rmsds], np.mean(_rmsds), np.std(_rmsds)
        )
    rmsds.append(_rmsds)

    with open(f"results/{filename}", "wb") as fo:
        pickle.dump(
            {
                "rmsd": rmsds,
            },
            fo,
        )
