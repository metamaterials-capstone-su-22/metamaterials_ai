from models.ann import ModelMaker as ann
from models.cnn import ModelMaker as cnn
from models.mlpmixer import ModelMaker as mlp_mixer
from models.resnet1d import ModelMaker as resnet_1d
from models.resnet2d import ModelMaker as resnet_2d
from models.res_ann import ModelMaker as resann

from .model_config import ModelConfig


class ModelArchFactory:
    """This class generate models based on requested architecture"""

    def create_model_arch(model_config: ModelConfig):
        if model_config.arch == "MLPMixer":
            return mlp_mixer.create_model(model_config)
        if model_config.arch == "resnet1d":
            return resnet_1d.create_model(model_config)
        if model_config.arch == "resnet2d":
            return resnet_2d.create_model(model_config)
        if model_config.arch == "ann":
            return ann.create_model(model_config)
        if model_config.arch == "cnn":
            return cnn.create_model(model_config)
        if model_config.arch == "res-ann":
            return resann.create_model(model_config)
