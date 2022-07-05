from distutils.file_util import copy_file

from config import Config

from .backward_model import InverseModel
from .forward_model import DirectModel


class ModelFactory:
    def __init__(self, config: Config) -> None:
        self.config = config

    def create_model(self, direction: str, direct_model: DirectModel = None):
        model = None
        config = self.config
        if direction == "inverse":
            model = InverseModel(config, direct_model)
        else:
            model = DirectModel(config)
        return model
