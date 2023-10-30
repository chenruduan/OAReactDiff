from typing import List, Optional, Tuple
from uuid import uuid4
import os
import shutil

from pl_trainer import ConfModule, DDPMModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from oa_reactdiff.trainer.ema import EMACallback
from oa_reactdiff.model import EGNN, LEFTNet, MACE


model_type = "leftnet"
version = "8"
project = "TSDiffusion-TS1x-All-Confidence"
# ---EGNNDynamics---
egnn_config = dict(
    in_node_nf=8,  # embedded dim before injecting to egnn
    in_edge_nf=0,
    hidden_nf=256,
    edge_hidden_nf=64,
    act_fn="swish",
    n_layers=9,
    attention=True,
    out_node_nf=None,
    tanh=True,
    coords_range=15.0,
    norm_constant=1.0,
    inv_sublayers=1,
    sin_embedding=True,
    normalization_factor=1.0,
    aggregation_method="mean",
)
leftnet_config = dict(
    pos_require_grad=False,
    cutoff=10.0,
    num_layers=6,
    hidden_channels=196,
    num_radial=96,
    in_hidden_channels=8,
    reflect_equiv=True,
    legacy=True,
    update=True,
    pos_grad=False,
    single_layer_output=True,
)
mace_config = dict(
    r_max=10.0,
    num_bessel=16,
    num_polynomial_cutoff=5,
    max_ell=3,
    num_interactions=3,
    in_node_nf=8,
    hidden_irreps="64x0e + 64x1o",
    MLP_irreps="64x0e + 64x1o",
    avg_num_neighbors=10.0,
    correlation=3,
    act_fn="silu",
    hidden_channels=64,
)

if model_type == "leftnet":
    model_config = leftnet_config
    model = LEFTNet
elif model_type == "egnn":
    model_config = egnn_config
    model = EGNN
elif model_type == "mace":
    model_config = mace_config
    model = MACE
else:
    raise KeyError("model type not implemented.")

optimizer_config = dict(
    lr=5e-4,
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)

T_0 = 200
T_mult = 2
training_config = dict(
    datadir="../data/transition1x/",
    remove_h=False,
    bz=8,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    ema=False,
    swapping_react_prod=True,
    append_frag=False,
    use_by_ind=True,
    reflection=False,
    single_frag_only=False,
    only_ts=False,
    confidence_model=False,
    ediff="reactant",
    position_key="positions",
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=20,
    ),  # step
)


node_nfs: List[int] = [9] * 3  # 3 (pos) + 5 (cat) + 1 (charge)
edge_nf: int = 0  # edge type
condition_nf: int = 1
fragment_names: List[str] = ["inorg_node", "org_edge", "org_node"]
pos_dim: int = 3
update_pocket_coords: bool = True
condition_time: bool = True
edge_cutoff: Optional[float] = None
loss_type = "l2"
pos_only = True
process_type = "TS1x"
enforce_same_encoding = None

run_name = f"{model_type}-{version}-" + str(uuid4()).split("-")[-1]

tspath = "/home/ubuntu/efs/TSDiffusion/oa_reactdiff/trainer/ckpt/TSDiffusion-TS1x-All"
ddpm_trainer = DDPMModule.load_from_checkpoint(
    checkpoint_path=f"{tspath}/leftnet-8-70b75beeaac1/ddpm-epoch=2074-val-totloss=531.18.ckpt",
    map_location="cpu",
)
source = {
    "model": ddpm_trainer.ddpm.dynamics.model.state_dict(),
    "encoders": ddpm_trainer.ddpm.dynamics.encoders.state_dict(),
    "decoders": ddpm_trainer.ddpm.dynamics.decoders.state_dict(),
}
# source = None


seed_everything(42, workers=True)
ddpm = ConfModule(
    model_config=model_config,
    optimizer_config=optimizer_config,
    training_config=training_config,
    node_nfs=node_nfs,
    edge_nf=edge_nf,
    condition_nf=condition_nf,
    fragment_names=fragment_names,
    pos_dim=pos_dim,
    edge_cutoff=edge_cutoff,
    process_type=process_type,
    model=model,
    enforce_same_encoding=enforce_same_encoding,
    source=source,
    classification=False,
    name_temp="hold_addprop",  # conf_hold for confidence
    target_key="ediff",
)

config = model_config.copy()
config.update(optimizer_config)
config.update(training_config)
trainer = None
if trainer is None or (isinstance(trainer, Trainer) and trainer.is_global_zero):
    wandb_logger = WandbLogger(
        project=project,
        log_model=False,
        name=run_name,
    )
    try:  # Avoid errors for creating wandb instances multiple times
        wandb_logger.experiment.config.update(config)
        wandb_logger.watch(ddpm.confidence, log="all", log_freq=100, log_graph=False)
    except:
        pass

ckpt_path = f"checkpoint/{project}/{wandb_logger.experiment.name}"
earlystopping = EarlyStopping(
    monitor="val-totloss",
    patience=2000,
    verbose=True,
    log_rank_zero_only=True,
)
checkpoint_callback = ModelCheckpoint(
    monitor="val-totloss",
    dirpath=ckpt_path,
    filename="ddpm-{epoch:03d}-{val-totloss:.4f}-{val-MAE:.4f}-{val-Pearson:.4f}",
    every_n_epochs=5,
    save_top_k=-1,
)
lr_monitor = LearningRateMonitor(logging_interval="step")
callbacks = [earlystopping, checkpoint_callback, TQDMProgressBar(), lr_monitor]
if training_config["ema"]:
    callbacks.append(EMACallback())

strategy = None
devices = [0]
# strategy = DDPStrategy(find_unused_parameters=True)
# if strategy is not None:
#     devices = [x for x in range(8)]
trainer = Trainer(
    max_epochs=3000,
    accelerator="gpu",
    deterministic=False,
    devices=devices,
    strategy=strategy,
    log_every_n_steps=10,
    callbacks=callbacks,
    profiler=None,
    logger=wandb_logger,
    accumulate_grad_batches=2,
    gradient_clip_val=training_config["gradient_clip_val"],
    limit_train_batches=0.1,
    limit_val_batches=1,
    # resume_from_checkpoint=f"./ckpt/{project}/leftnet-8-2b217103c3fb/ddpm-epoch=004-val-totloss=811.47.ckpt",
    # max_time="00:10:00:00",
)

trainer.fit(ddpm)
