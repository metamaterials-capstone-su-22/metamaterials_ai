from config import Config
from trainer_factory import TrainerFactory
from utils import DataUtils, FileUtils
from pathlib import Path
import os

def main(config: Config):
    # TODO: add step to get raw file and create pt file from them (raw2pt)
    # Get pt files (Downloading them if they are missing)
    FileUtils.get_pt_files(config)

    # Load pt file data
    data = DataUtils.read_pt_data(config)
    forward_model = None
    # 2. forward training_step
    if config.use_forward:
        forward_trainer = TrainerFactory(config, data, "forward")
        if config.load_forward_checkpoint:
            forward_trainer.test(
                checkpoint_path=str(
                    max(
                        Path(f"{config.work_path}/weights/forward").glob("*.ckpt"),
                        key=os.path.getctime,
                    )
                )
            )
        else:
            forward_trainer.fit()
        forward_model = forward_trainer.model

    # 3. backward training
    backward_trainer = TrainerFactory(config, data, 'backward', forward_model)
    backward_trainer.fit()

    # 4. visualization

    return True


if __name__ == "__main__":
    config = Config()
    main(config)
