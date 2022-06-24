from models.mlpmixer import ModelMaker as mlpmixermaker
from src.models.ANN import model_maker
from .model_config import ModelConfig


class ModelArchFactory:
    """This class generate models based on requested architecure"""

    def create_model_arch(model_config: ModelConfig):
        if model_config.arch == "MLPMixer":
            return mlpmixermaker.create_model(model_config)
        if model_config.arch == "resnet1d":
            return model_maker.create_model(model_config)
            
