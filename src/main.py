from config import Config
from data_modules import ForwardDataModule
from models import Model, ModelConfig
from trainer import Trainer
from utils import DataUtils, FileUtils


def main(config: Config):
    # TODO: add step to get raw file and create pt file from them (raw2pt)
    # Get pt files (Downloading them if they are missing)
    FileUtils.get_pt_files(config)

    # Load pt file data
    data = DataUtils.read_pt_data(config)

    # 2. forward training_step

    # 2.1 Model Architecture
    # Note different arch can be loaded
    forward_model_config = ModelConfig(
        arch=config.model_arch, direction="forward", num_classes=config.num_wavelens
    )
    forward_trainer = Trainer(config, "forward").get_trainer()
    forward_data_module = ForwardDataModule(config, data, "forward")
    forward_model = Model(config, forward_model_config)
    forward_trainer.fit(model=forward_model, datamodule=forward_data_module)

    # 3. backward training
    # backward_model_config = ModelConfig(
    #     arch=config.arch
    #     , direction='backward'
    #     , in_channels=config.num_wavelens)
    # backward_trainer = Trainer(config, 'backward').get_trainer()
    # backward_data_module = ForwardDataModule(config, data, 'backward')

    # 4. visualization

    return True


if __name__ == "__main__":
    config = Config()
    main(config)
