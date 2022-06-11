from models.mplmixer import ModelMaker

from .model_config import ModelConfig


class ModelArchFactory:
    """This class generate models based on requested architecure"""

    def create_model_arch(model_config: ModelConfig):
        if model_config.arch == "MLPMixer":
            return ModelMaker.create_model(model_config)
