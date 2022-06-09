from .mlp_mixer import MLPMixerArc
from .model_config import ModelConfig

class ModelFactory():
    @staticmethod
    def get_model(model_config: ModelConfig):
        if model_config.arch == 'MLPMixer':
            return MLPMixerArc.create_model(model_config)
