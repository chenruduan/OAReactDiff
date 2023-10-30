from typing import List
import time
import os
import numpy as np
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
from oa_reactdiff.analyze.geomopt import calc_deltaE, compute_efh
from oa_reactdiff.evaluate.utils import (
    set_new_schedule,
    inplaint_batch,
    batch_ts_deltaE,
)
from oa_reactdiff.utils.sampling_tools import write_tmp_xyz

EV2KCALMOL = 23.06
AU2KCALMOL = 627.5


def save_pickle(filename, rmsds, deltaEs):
    with open(f"results/{filename}", "wb") as fo:
        pickle.dump(
            {
                "rmsd": rmsds,
                "ts_deltaE": deltaEs,
            },
            fo,
        )


config = dict(
    timesteps=150,
    bz=16,
    resamplings=5,
    jump_length=5,
    repeats=1,
    max_batch=-1,
    shuffle=False,
    single_frag_only=False,
)


filename = ""
for k, v in config.items():
    filename += f"rmsdvsEdiff_{k}-{v}_"
filename += ".pkl"
print(filename)

print("loading ddpm trainer...")
device = torch.device("cuda")
tspath = "/home/ubuntu/efs/2Dto3D_ReactGen/oa_reactdiff/trainer/ckpt/TSDiffusion-TS1x"
checkpoints = {
    "leftnet_2074": f"{tspath}-All/leftnet-8-70b75beeaac1/ddpm-epoch=2074-val-totloss=531.18.ckpt",
}
ddpm_trainer = DDPMModule.load_from_checkpoint(
    checkpoint_path=checkpoints["leftnet_2074"],
    map_location=device,
)
ddpm_trainer = set_new_schedule(
    ddpm_trainer,
    timesteps=config["timesteps"],
    noise_schedule="polynomial_2",
)

print("loading dataset...")
dataset = ProcessedTS1x(
    npz_path="../data/transition1x/valid.pkl",
    center=True,
    pad_fragments=0,
    device="cuda",
    zero_charge=False,
    remove_h=False,
    single_frag_only=config["single_frag_only"],
    swapping_react_prod=False,
    use_by_ind=True,
)
loader = DataLoader(
    dataset,
    batch_size=config["bz"],
    shuffle=config["shuffle"],
    num_workers=0,
    collate_fn=dataset.collate_fn,
)
_id = uuid4()
localpath = "tmp/" + str(_id)
os.makedirs(localpath)

print("evaluating...")
rmsds, deltaEs = [], []
TSEs = []
for num_repeat in range(config["repeats"]):
    print("num_repeat: ", num_repeat)
    _rmsds, _deltaEs = [], []
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
        )
        write_tmp_xyz(fragments_nodes, out_samples, idx=[0, 1, 2], localpath=localpath)
        write_tmp_xyz(
            fragments_nodes, xh_fixed, idx=[1], prefix="sample", localpath=localpath
        )
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

        _deltaEs += batch_ts_deltaE(config["bz"], xc="wb97x", localpath=localpath)
        print("deltaEs: ", [round(_x, 2) for _x in _deltaEs], np.mean(np.abs(_deltaEs)))
        save_pickle(filename, _rmsds, _deltaEs)
    rmsds.append(_rmsds)
    deltaEs.append(_deltaEs)
save_pickle(filename, rmsds, deltaEs)
