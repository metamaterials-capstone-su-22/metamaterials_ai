from data_module import DataModule
from meta_trainer import MetaTrainer
from models import ModelFactory
from models.forward_model import DirectModel
from trainer_factory import TrainerFactory


class MetaTrainerFactory:
    def __init__(self, config) -> None:
        self.config = config
        self.trainer_factory = TrainerFactory(config)
        self.model_factory = ModelFactory(config)

    def create_meta_trainer(self, direction, direct_model: DirectModel = None):
        config = self.config
        trainer = self.trainer_factory.create_trainer(direction)
        model = self.model_factory.create_model(direction, direct_model)
        data_module = DataModule(config, direction)

        return MetaTrainer(
            config=config,
            direction=direction,
            model=model,
            data_module=data_module,
            trainer=trainer,
        )
