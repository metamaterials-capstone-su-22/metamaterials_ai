from config import Config
from utils import FileUtils, DataUtils
from trainer import Trainer
from data_modules import ForwardDataModule
from models import ForwardModel

def main(config):
    # TODO: add step to get raw file and create pt file from them (raw2pt)
    # Get pt files (Downloading them if they are missing)
    FileUtils.get_pt_files(config)

    # Load pt file data
    data = DataUtils.read_pt_data(config)

    # 2. forward training_step
    forward_trainer = Trainer(config, 'forward').get_trainer()
    forward_data_module = ForwardDataModule(config, data)
    forward_model = ForwardModel(config)
    forward_trainer.fit(model=forward_model, datamodule=forward_data_module)

    # 3. backward training
    # 4. visualization

    return True


if __name__ == "__main__":
    config = Config()
    main(config)
