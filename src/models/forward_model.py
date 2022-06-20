#!/usr/bin/env python3
from __future__ import annotations

import torch

from config import Config
from utils import rmse

from .base_model import BaseModel
from .model_arch_factory import ModelArchFactory
from .model_config import ModelConfig


class ForwardModel(BaseModel):
    def __init__(self, config: Config):
        self.model_config = ModelConfig(
            arch=config.model_arch, direction="forward", num_classes=config.num_wavelens
        )
        super().__init__(config, direction="forward")
        self.save_hyperparameters(config.__dict__)


    def create_model_arc(self):
        return ModelArchFactory.create_model_arch(self.model_config)

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
