from distutils.file_util import copy_file

from config import Config

from .backward_model import BackwardModel
from .forward_model import ForwardModel


class ModelFactory:
    def __init__(self, config: Config) -> None:
        self.config = config

    def create_model(self, direction: str, forward_model: ForwardModel = None):
        model = None
        config = self.config
        if direction == "backward":
            model = BackwardModel(config, forward_model)
        else:
            model = ForwardModel(config)
        return model
