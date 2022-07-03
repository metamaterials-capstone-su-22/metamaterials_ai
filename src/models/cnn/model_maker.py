from einops.layers.torch import Rearrange
from torch import nn

from models.model_config import ModelConfig

from .module import BlocksBuilder


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
        size = 1
        return nn.Sequential(
            Rearrange("b c -> b c 1"),
            nn.Conv1d(
                in_channels=model_config.in_channels, out_channels=dim, kernel_size=size
            ),
            nn.BatchNorm1d(dim),
            nn.SELU(),
            BlocksBuilder(num_blocks=num_blocks, dim=dim),
            nn.Conv1d(
                in_channels=dim, out_channels=model_config.num_classes, kernel_size=size
            ),
            nn.SELU(),
            Rearrange("b c h -> (b h) c"),
            nn.Sigmoid(),
        )

    @staticmethod
    def create_inverse_model(model_config: ModelConfig):
        num_blocks: int = 16
        dim: int = 1_200
        kernel_size = 1
        return nn.Sequential(
            Rearrange("b c -> b c 1"),
            nn.Conv1d(
                in_channels=model_config.in_channels,
                out_channels=dim,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm1d(dim),
            nn.SELU(),
            BlocksBuilder(num_blocks=num_blocks, dim=dim),
            nn.Conv1d(
                in_channels=dim,
                out_channels=model_config.num_classes,
                kernel_size=kernel_size,
            ),
            nn.SELU(),
            Rearrange("b c h -> (b h) c"),
            nn.Flatten(),
        )
