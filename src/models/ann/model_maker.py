from einops.layers.torch import Rearrange
from torch import nn

from models.model_config import ModelConfig


class ModelMaker:
    @staticmethod
    def create_model(model_config: ModelConfig):
        return (
            ModelMaker.create_forward_model(model_config)
            if model_config.direction == "forward"
            else ModelMaker.create_backward_model(model_config)
        )

    @staticmethod
    def create_forward_model(model_config: ModelConfig):
        return nn.Sequential(
            nn.Linear(14, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 264),
            nn.Linear(264, 512),
            nn.Linear(512, 800),
            nn.Sigmoid(),
        )
