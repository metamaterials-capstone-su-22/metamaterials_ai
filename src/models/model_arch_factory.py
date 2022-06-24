from models.mlpmixer import ModelMaker as mlpmixermaker
from src.models.resnet1d import ModelMaker as resnetmaker
from .model_config import ModelConfig


class ModelArchFactory:
    """This class generate models based on requested architecure"""

    def create_model_arch(model_config: ModelConfig):
        if model_config.arch == "MLPMixer":
            return mlpmixermaker.create_model(model_config)
        if model_config.arch == "resnet1d":
            return resnetmaker.create_model(model_config)
            
