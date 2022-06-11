from data_module import BackwardDataModule, ForwardDataModule
from meta_trainer import MetaTrainer
from models import ModelFactory
from src.models.forwards import ForwardModel
from trainer_factory import TrainerFactory


class MetaTrainerFactory:
    """This class wraps the model, trainer, and datamodule
    for each direction (forward and backward)"""

    def __init__(self, config) -> None:
        self.config = config
        self.trainer_factory = TrainerFactory(config)
        self.model_factory = ModelFactory(config)

    def create_meta_trainer(self, direction, forward_model: ForwardModel = None):
        config = self.config
        trainer = self.trainer_factory.create_trainer(direction)
        model = self.model_factory.create_model(direction, forward_model)
        data_module = None
        if direction == "backward":
            data_module = BackwardDataModule(config)
        else:
            data_module = ForwardDataModule(config)

        return MetaTrainer(
            config=config,
            direction=direction,
            model=model,
            data_module=data_module,
            trainer=trainer,
        )
