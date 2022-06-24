from einops.layers.torch import Rearrange
from torch import nn

from models.model_config import ModelConfig

from .net1d_module import RESNET1D


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
        RESNET1D(
        in_channels=model_config.in_channels, 
        base_filters=64, 
        kernel_size=16, 
        stride=2, 
        groups=4, 
        n_block=4, 
        n_classes=model_config.num_classes, 
        downsample_gap=2, 
        increasefilter_gap=4, 
        use_bn=True, 
        use_do=True, 
        verbose=False),
        nn.Sigmoid()
        )

    @staticmethod
    def create_backward_model(model_config: ModelConfig):
        return nn.Sequential(
        RESNET1D(
        in_channels=model_config.in_channels, 
        base_filters=64, 
        kernel_size=16, 
        stride=2, 
        groups=4, 
        n_block=4, 
        n_classes=model_config.num_classes, 
        downsample_gap=2, 
        increasefilter_gap=4, 
        use_bn=True, 
        use_do=True, 
        verbose=False),
        nn.Flatten())
