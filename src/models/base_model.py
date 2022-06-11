from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

import nngraph
from config import Config
from mixer import MLPMixer
from utils import Stage, rmse, split


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        config: Config,
        direction: str,
    ) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.config = config
        self.model = self.create_model_arc()
        self.initialize_model()
        self.work_path = config.work_path
        self.direction = direction

    def should_create_graph(self, stage):
        should_create_graph: bool = False
        # Note logig is from legacy code! (PM)
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
                    y_pred,
                    y,
                    f"{self.work_path}/{self.direction}_{stage}_points.txt",
                    all_points=True,
                )
            except Exception as e:
                print(f"Failed to save nngraph: {e}")
                raise

        self.log(f"{self.direction}/{stage}/loss", loss, prog_bar=True)