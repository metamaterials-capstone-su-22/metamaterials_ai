#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import torch
import wandb

from config import Config
from meta_trainer_factory import MetaTrainerFactory
from models import ForwardModel
from plotter import Plotter
from utils import FileUtils


def save_and_plot(
    backward_trainer, forward_model: ForwardModel, work_folder: str, data_folder: str
):
    preds: dict = backward_trainer.predict()

    torch.save(
        preds,
        Path(f"{work_folder}/preds.pt"),
    )
    wandb.finish()
    # plotter needs the forward model to plot the result.
    if forward_model:
        Plotter.plot_results(
            preds, forward_model, backward_trainer.model, work_folder, data_folder
        )


def train_backward(meta_trainer, forward_model):
    print("=" * 80)
    print("Backward Model Step")
    print("=" * 80)
    backward_trainer = meta_trainer.create_meta_trainer(
        "backward", forward_model)
    if not config.load_backward_checkpoint:
        backward_trainer.fit()

    backward_trainer.test()
    save_and_plot(
        backward_trainer, forward_model, config.work_folder, config.data_folder
    )


def setup():
    FileUtils.setup_folder_structure(config.work_folder, config.data_folder)
    FileUtils.fetch_pt_files(config.data_folder, config.data_file)


def main(config: Config) -> None:
    setup()
    meta_trainer = MetaTrainerFactory(config)
    forward_model = None
    if config.direction in ['direct', 'both']:
        if config.use_forward:
            forward_trainer = meta_trainer.create_meta_trainer("forward")
            if not config.load_forward_checkpoint:
                forward_trainer.fit()
            forward_trainer.test()
            forward_model = forward_trainer.model

    # Close the Forward before backward if you want separate project
    wandb.finish()
    if config.direction in ['inverse', 'both']:
        train_backward(meta_trainer, forward_model)


if __name__ == "__main__":
    config = Config()
    main(config)
