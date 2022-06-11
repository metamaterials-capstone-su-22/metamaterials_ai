#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import torch
import wandb

from config import Config
from file_utils import FileUtils
from meta_trainer_facotry import MetaTrainerFactory
from models import ForwardModel
from plotter import Plotter


def save_and_plot(backward_trainer, forward_model: ForwardModel):
    preds: dict = backward_trainer.predict()

    torch.save(
        preds,
        Path(f"{config.work_path}/preds.pt"),
    )
    wandb.finish()
    # plotter needs the forward model to plot the result.
    if forward_model:
        plotter = Plotter()
        plotter.plot_results(preds, forward_model, backward_trainer.model)


def train_backward(meta_trainer, forward_model):
    print("=" * 80)
    print("Backward Model Step")
    print("=" * 80)
    backward_trainer = meta_trainer.create_meta_trainer("backward", forward_model)
    if not config.load_backward_checkpoint:
        backward_trainer.fit()

    backward_trainer.test()
    save_and_plot(backward_trainer, forward_model)


def setup():
    FileUtils.setup_folder_structure(config.work_path, config.data_path)
    FileUtils.get_pt_files(config.data_path, config.data_file)


def main(config: Config) -> None:
    setup()
    meta_trainer = MetaTrainerFactory(config)
    forward_model = None
    if config.use_forward:
        forward_trainer = meta_trainer.create_meta_trainer("forward")
        if not config.load_forward_checkpoint:
            forward_trainer.fit()
        forward_trainer.test()
        forward_model = forward_trainer.model

    train_backward(meta_trainer, forward_model)


if __name__ == "__main__":
    config = Config()
    main(config)
