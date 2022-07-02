from __future__ import annotations
from distutils.command.config import config

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn, optim

import nngraph
from config import Config
from utils import Stage, rmse, split


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        config: Config,
        direction: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.direction = direction
        self.work_folder = config.work_folder
        self.model = self.create_model_arc()
        self.initialize_model()

    def should_create_graph(self, stage):
        should_create_graph: bool = False
        # Note loging is from legacy code! (PM)
        if stage == "tarin":
            should_create_graph = (
                self.current_epoch == self.config.forward_num_epochs - 5
            )
        elif stage == "val":
            should_create_graph = (
                self.current_epoch > self.config.forward_num_epochs - 5
            )
        else:  # for Test it is always ture
            should_create_graph = True
        return should_create_graph

    def create_graph_and_log(self, stage, y_pred, y, loss):
        if self.config.create_plots and self.should_create_graph(stage):
            try:
                nngraph.save_integral_emiss_point(
                    self.config,
                    y_pred,
                    y,
                    f"{self.work_folder}/{self.direction}_{stage}_points.txt",
                    all_points=True,
                )
            except Exception as e:
                print(f"Failed to save nngraph: {e}")
                raise

        self.log(f"{self.direction}/{stage}/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr,
                                amsgrad=True,
                                weight_decay=self.config.weight_decay)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 50, 150, 300], gamma=0.1)

        return [optimizer], [lr_scheduler]
