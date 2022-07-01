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
            nn.Linear(14, 128),
            nn.SELU(),
            nn.Linear(128, 264),
            nn.SELU(),
            nn.Linear(264, 512),
            nn.SELU(),
            nn.Linear(512, 512),
            nn.SELU(),
            nn.Linear(512, 512),
            nn.SELU(),
            nn.Linear(512, 800),
            nn.Sigmoid(),
        )
