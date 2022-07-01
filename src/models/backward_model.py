from __future__ import annotations

import torch
from typing import Optional
from config import Config
from utils import rmse
from .forward_model import ForwardModel
from .base_model import BaseModel
from .model_arch_factory import ModelArchFactory
from .model_config import ModelConfig


class BackwardModel(BaseModel):
    def __init__(self, config: Config,  forward_model: Optional[ForwardModel] = None,):
        self.model_config = ModelConfig(
            arch=config.model_arch,
            direction="backward",
            num_classes=14,
            in_channels=config.num_wavelens
        )
        super().__init__(config, direction="backward")
        self.save_hyperparameters(config.__dict__)

    def create_model_arc(self):
        return ModelArchFactory.create_model_arch(self.model_config)

    def initialize_model(self):
        # This call *must* happen to initialize the lazy layers
        self.forward(torch.rand(2,  self.model_config.in_channels))

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
