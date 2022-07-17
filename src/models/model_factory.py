from distutils.file_util import copy_file

from config import Config

from .direct_model import DirectModel
from .inverse_model import InverseModel


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
