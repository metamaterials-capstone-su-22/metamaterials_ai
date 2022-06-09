#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import random
import sys
from cProfile import label
from pathlib import Path
from typing import List, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from backwards import BackwardModel
from data import BackwardDataModule, ForwardDataModule, StepTestDataModule
from einops import rearrange, reduce, repeat
from forwards import ForwardModel
from nngraph import graph
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from ray import tune
from sklearn.metrics import mean_squared_error
from torch import Tensor

import utils
from utils import Config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--forward-num-epochs",
    "--fe",
    type=int,
    default=None,
    help="Number of epochs for forward model",
)
parser.add_argument(
    "--num-wavelens",
    type=int,
    default=800,
    help="Number of wavelens to interpolate to",
)
parser.add_argument(
    "--backward-num-epochs",
    "--be",
    type=int,
    default=None,
    help="Number of epochs for backward model",
)
parser.add_argument(
    "--forward-batch-size",
    "--fbs",
    type=int,
    default=None,
    help="Batch size for forward model",
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=1,
    help="How many runs for optimization",
)
parser.add_argument(
    "--backward-batch-size",
    "--bbs",
    type=int,
    default=None,
    help="Batch size for backward model",
)
parser.add_argument(
    "--prediction-iters",
    type=int,
    default=1,
    help="Number of iterations to run predictions",
)
parser.add_argument(
    "--use-cache",
    type=eval,
    choices=[True, False],
    default=False,
    help="Load saved dataset (avoids 1 minute startup cost of fetching data from database, useful for quick tests).",
)
parser.add_argument(
    "--use-forward",
    type=eval,
    choices=[True, False],
    default=True,
    help="Whether to use a forward model at all",
)
parser.add_argument(
    "--load-forward-checkpoint",
    type=eval,
    choices=[True, False],
    default=False,
    help="Load trained forward model. Useful for validation. Requires model to already be trained and saved.",
)
parser.add_argument(
    "--load-backward-checkpoint",
    type=eval,
    choices=[True, False],
    default=False,
    help="Load trained backward model. Useful for validation. Requires model to already be trained and saved.",
)
args = parser.parse_args()

# The `or` idiom allows overriding values from the command line.
config: Config = {
    # "forward_lr": tune.loguniform(1e-7, 1e-4),
    "forward_lr": 1e-6,
    "backward_lr": tune.loguniform(1e-6, 1e-5),
    "forward_num_epochs": args.forward_num_epochs or tune.choice([1600]),
    # args.backward_num_epochs or tune.choice([2500]),
    "backward_num_epochs": 2,
    "forward_batch_size": args.forward_batch_size or tune.choice([2**9]),
    "backward_batch_size": args.backward_batch_size or tune.choice([2**9]),
    "use_cache": True,
    # "use_cache": args.use_cache,
    "use_forward": args.use_forward,
    "load_forward_checkpoint": True,
    "load_backward_checkpoint": True,
    "num_wavelens": args.num_wavelens,
    # Path to the working folder, checkpoint, graphs, ..
    "work_path": "local_work/",
    "data_path": "local_data/",  # Path to the data folder
}

concrete_config: Config = Config(
    {k: (v.sample() if hasattr(v, "sample") else v) for k, v in config.items()}
)


def main(config: Config) -> None:

    forward_trainer = pl.Trainer(
        max_epochs=config["forward_num_epochs"],
        logger=[
            WandbLogger(
                name="Forward laser params",
                save_dir="/data-new/spencersong/wandb_logs/forward",
                offline=False,
                project="Laser Forward",
                log_model=True,
            ),
            TensorBoardLogger(
                save_dir="/data-new/spencersong/test_tube_logs/forward",
                name="Forward",
            ),
        ],
        callbacks=[
            ModelCheckpoint(
                monitor="forward/val/loss",
                dirpath="/data-new/spencersong/weights/forward",
                save_top_k=1,
                mode="min",
                save_last=True,
            ),
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=2),
        ],
        gpus=1,
        precision=32,
        # overfit_batches=1,
        # track_grad_norm=2,
        weights_summary="full",
        check_val_every_n_epoch=min(3, config["forward_num_epochs"] - 1),
        gradient_clip_val=0.5,
        log_every_n_steps=min(3, config["forward_num_epochs"] - 1),
    )

    backward_trainer = pl.Trainer(
        max_epochs=config["backward_num_epochs"],
        logger=[
            WandbLogger(
                name="Backward laser params",
                save_dir="/data-new/spencersong/wandb_logs/backward",
                offline=False,
                project="Laser Backward",
                log_model=True,
            ),
            TensorBoardLogger(
                save_dir="/data-new/spencersong/test_tube_logs/backward",
                name="Backward",
            ),
        ],
        callbacks=[
            ModelCheckpoint(
                monitor="backward/val/loss",
                dirpath="/data-new/spencersong/weights/backward",
                save_top_k=1,
                mode="min",
                save_last=True,
            ),
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=10),
        ],
        gpus=1,
        precision=32,
        weights_summary="full",
        check_val_every_n_epoch=min(3, config["backward_num_epochs"] - 1),
        gradient_clip_val=0.5,
        log_every_n_steps=min(3, config["backward_num_epochs"] - 1),
    )

    forward_data_module = ForwardDataModule(config)
    backward_data_module = BackwardDataModule(config)

    # TODO: load checkpoint for both forward and back
    if config["use_forward"]:
        forward_model = ForwardModel(config)
        if not config["load_forward_checkpoint"]:
            forward_trainer.fit(model=forward_model, datamodule=forward_data_module)

        forward_trainer.test(
            model=forward_model,
            ckpt_path=str(
                max(
                    Path("/data-new/spencersong/weights/forward").glob("*.ckpt"),
                    key=os.path.getctime,
                )
            ),
            datamodule=forward_data_module,
        )
        backward_model = BackwardModel(config=config, forward_model=forward_model)
    else:
        backward_model = BackwardModel(config=config, forward_model=None)

    if not config["load_backward_checkpoint"]:
        backward_trainer.fit(model=backward_model, datamodule=backward_data_module)

    backward_trainer.test(
        model=backward_model,
        ckpt_path=str(
            max(
                Path("/data-new/spencersong/weights/backward").glob("*.ckpt"),
                key=os.path.getctime,
            )
        ),
        datamodule=backward_data_module,
    )

    preds: dict = backward_trainer.predict(
        model=backward_model,
        ckpt_path=str(
            max(
                Path("/data-new/spencersong/weights/backward").glob("*.ckpt"),
                key=os.path.getctime,
            )
        ),
        datamodule=backward_data_module,
        return_predictions=True,
    )
    torch.save(
        preds,
        Path(f"/data-new/spencersong/preds.pt"),
    )
    wandb.finish()

    # graph(residualsflag = True, predsvstrueflag = True, index_str = params_str, target_str = save_str)
    train_emiss = torch.load("/data-new/spencersong/data.pt")["interpolated_emissivity"]

    true_emiss = preds[0]["true_emiss"]
    pred_array = []

    # Variant num is the number of random curves to generate with jitter
    variant_num = 1
    # Arbitrary list is the indices you want to look at in a tensor of emissivity curves. In the FoMM case, 0 = cutoff at 2.5 wl, 800 = cutoff at 12.5 wl.
    arbitrary_list = [220]
    watt_list = [[] for i in range(variant_num)]
    speed_list = [[] for i in range(variant_num)]
    spacing_list = [[] for i in range(variant_num)]
    random_emissivities_list = [[] for i in range(variant_num)]
    param_std_total = 0
    print("start")
    for i in range(variant_num):
        # new_true = [torch.tensor(emiss+random.uniform(-0.05, 0.05)) for emiss in true_emiss]
        # jitter isn't doing anything XXX
        random_mult = random.uniform(-0.3, 0.3)
        new_true = torch.clamp(
            torch.tensor(
                [
                    [
                        (random_mult * (1 / emiss) * (e_index / 3 + 100) / 600) * emiss
                        + emiss
                        for e_index, emiss in enumerate(sub_emiss)
                    ]
                    for sub_emiss in true_emiss
                ]
            ),
            0,
            1,
        )

        if i == 0:
            new_true = true_emiss
        back = backward_model(new_true)
        # add spacing

        # minspeed = 10, maxspeed = 700

        # min 1 max 42

        new_pred = forward_model(back)

        pred_array.append(new_pred.detach())

    for i in arbitrary_list:

        pred_emiss = []
        for j in range(variant_num):
            pred_emiss.append(pred_array[j][i])
        pred_emiss = torch.stack(pred_emiss)
        fig = plot_val(pred_emiss, true_emiss[i], i)
        fig.savefig(f"/data-new/spencersong/figs/{i}_predicted.png", dpi=300)
        plt.close(fig)


def plot_val(pred_emiss, true_emiss, index):
    wavelen = torch.load("/data-new/spencersong/data.pt")["interpolated_wavelength"][0]
    pred_emiss = pred_emiss[0]
    extended_max = 2.5
    extended_min = 0.1

    granularity = 192

    extension = torch.tensor(
        [
            extended_min + (i) / granularity * (extended_max - extended_min)
            for i in range(granularity)
        ]
    )

    extended_wave = torch.cat((extension, wavelen))

    # extend the pred emiss
    old_emiss = pred_emiss
    first_emiss = np.float(old_emiss[0])
    new_emiss = torch.cat(
        (torch.tensor([first_emiss for i in range(granularity)]), old_emiss)
    )
    pred_emiss = new_emiss

    # extend the true emiss
    old_emiss = true_emiss
    first_emiss = np.float(old_emiss[0])
    new_emiss = torch.cat(
        (torch.tensor([first_emiss for i in range(granularity)]), old_emiss)
    )
    true_emiss = new_emiss

    wavelen = extended_wave

    fig, ax = plt.subplots()
    temp = 1400
    plot_index = 0
    planck = [float(utils.planck_norm(wavelength, temp)) for wavelength in wavelen]

    planck_max = max(planck)
    planck = [wave / planck_max for wave in planck]

    new_score = 0

    wavelen_cutoff = float(wavelen[index + granularity])
    # format the predicted params
    FoMM = utils.planck_emiss_prod(wavelen, pred_emiss, wavelen_cutoff, 1400)

    ax.plot(
        wavelen,
        pred_emiss,
        c="blue",
        alpha=0.2,
        linewidth=1.0,
        label=f"Predicted Emissivity, FoMM = {FoMM}",
    )
    ax.plot(
        wavelen, true_emiss, c="black", label=f"Ideal target emissivity", linewidth=2.0
    )
    ax.legend()
    return fig


if __name__ == "__main__":
    main(concrete_config)
