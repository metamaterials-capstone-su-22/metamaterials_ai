#!/usr/bin/env python3

from __future__ import annotations

import random
from pathlib import Path

import torch
import wandb

from config import Config
from meta_trainer_factory import MetaTrainerFactory
from models import DirectModel
from plotter import Plotter
from utils import FileUtils


def save_and_plot(
    inverse_trainer, direct_model: DirectModel, work_folder: str, data_folder: str
):
    preds: dict = inverse_trainer.predict()

    torch.save(
        preds,
        Path(f"{work_folder}/preds.pt"),
    )
    wandb.finish()
    # plotter needs the forward model to plot the result.
    if direct_model:
        Plotter.plot_results(preds, direct_model,
                             inverse_trainer.model, config)


def train_inverse(meta_trainer, direct_model):
    print("=" * 80)
    print("Inverse Model Step")
    print("=" * 80)
    inverse_trainer = meta_trainer.create_meta_trainer(
        "inverse", direct_model)
    if not config.load_inverse_checkpoint:
        inverse_trainer.fit()
        FileUtils.save_best_model(config.work_folder, inverse_trainer)

    inverse_trainer.test()
    save_and_plot(
        inverse_trainer, direct_model, config.work_folder, config.data_folder
    )


def setup():
    # Fix the seed
    random.seed(100)
    FileUtils.setup_folder_structure(config.work_folder, config.data_folder)
    FileUtils.fetch_pt_files(config.data_folder, config.data_file)


def main(config: Config) -> None:
    setup()
    meta_trainer = MetaTrainerFactory(config)
    direct_model = None
    if config.direction in ["direct", "both"]:
        if config.use_direct:
            direct_trainer = meta_trainer.create_meta_trainer("direct")
            if not config.load_direct_checkpoint:
                direct_trainer.fit()
                FileUtils.save_best_model(config.work_folder, direct_trainer)

            direct_trainer.test()
            direct_model = direct_trainer.model

    # Close the Direct before Inverse if you want separate project
    wandb.finish()
    if config.direction in ["inverse", "both"]:
        train_inverse(meta_trainer, direct_model)


if __name__ == "__main__":
    config = Config()
    main(config)
