from torch import nn

from models.model_config import ModelConfig
from .ann_module import BlocksBuilder


class ModelMaker:
    @staticmethod
    def create_model(model_config: ModelConfig):
        return (
            ModelMaker.create_direct_model(model_config)
            if model_config.direction == "forward"
            else ModelMaker.create_inverse_model(model_config)
        )

    @staticmethod
    def create_direct_model(model_config: ModelConfig):
        num_blocks: int = 16
        dim: int = 512
        return nn.Sequential(
            nn.Linear(model_config.in_channels, dim),
            nn.SELU(),
            BlocksBuilder(num_blocks=num_blocks, dim=dim),
            nn.Linear(dim, model_config.num_classes),
            nn.Sigmoid(),
        )

    @staticmethod
    def create_inverse_model(model_config: ModelConfig):
        num_blocks: int = 16
        dim: int = 1_200
        return nn.Sequential(
            nn.Linear(model_config.in_channels, dim),
            nn.SELU(),
            BlocksBuilder(num_blocks=num_blocks, dim=dim),
            nn.Linear(dim, model_config.num_classes),
            nn.Flatten(),
        )
