from __future__ import annotations

import torch

from config import Config
from utils import rmse

from .base_model import BaseModel
from .model_arch_factory import ModelArchFactory
from .model_config import ModelConfig


class DirectModel(BaseModel):
    def __init__(self, config: Config):
        config = Config.parse_obj(config) if type(config) is dict else config
        self.model_config = ModelConfig(
            arch=config.direct_arch,
            direction="direct",
            num_classes=config.num_wavelens,
            in_channels=14,
        )
        super().__init__(config, direction="direct")
        self.lr = config.direct_lr
        self.gamma = config.direct_gamma
        self.milestones = self.get_milestones(config.direct_milestones)
        self.example_input_array = torch.randn(1, 14)
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
