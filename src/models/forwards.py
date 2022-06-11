#!/usr/bin/env python3
from __future__ import annotations

import torch
from einops.layers.torch import Rearrange
from torch import nn

from config import Config
from mixer import MLPMixer
from utils import rmse

from .base_model import BaseModel


class ForwardModel(BaseModel):
    def __init__(self, config: Config):
        self.save_hyperparameters()
        super().__init__(config, direction="forward")

    def create_model_arc(self):
        return nn.Sequential(
            Rearrange("b c -> b c 1 1"),
            MLPMixer(
                in_channels=14,
                image_size=1,
                patch_size=1,
                num_classes=self.config.num_wavelens,
                dim=512,
                depth=8,
                token_dim=256,
                channel_dim=2048,
                dropout=0.5,
            ),
            nn.Sigmoid(),
        )

    def initialize_model(self):
        # TODO how to reverse the *data* in the Linear layers easily? transpose?
        # This call *must* happen to initialize the lazy layers
        self.forward(torch.rand(2, 14))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        return self.execute_step(batch, "train")

    def validation_step(self, batch, batch_nb):
        return self.execute_step(batch, "val")

    def test_step(self, batch, batch_nb):
        return self.execute_step(batch, "test")

    def execute_step(self, batch, stage):
        x, y, uids = batch
        y_pred = self(x)
        loss = rmse(y_pred, y)
        self.create_graph_and_log(stage, y_pred, y, loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.forward_lr)
