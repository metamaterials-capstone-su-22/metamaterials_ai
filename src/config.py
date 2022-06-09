from pydantic import BaseModel

from argparser import ArgParser


class Config(BaseModel):
    backward_batch_size: int = 200  # 2**9
    backward_lr: float = 1e-6  # tune.loguniform(1e-6, 1e-5),
    backward_num_epochs: int = 5  # 2500
    data_path: str = "local_data"  # Path to the data folder
    forward_batch_size: int = 200  # 2**9
    forward_lr: float = 1e-6  # tune.loguniform(1e-7, 1e-4),
    forward_num_epochs: int = 2  # 1600
    load_forward_checkpoint: bool = True
    load_backward_checkpoint: bool = True
    model_arch = "MLPMixer"  # options 'MLPMixer'
    num_gpu: int = 1  # number of GPU
    num_wavelens: int | None  # This will be set @ load time. ex. 800
    substrate: str = "stainless_steel"  # options "stainless_steel" , "inconel"
    use_cache: bool = True
    use_forward: bool = True
    # Path to the working folder, checkpoint, graphs, ..
    work_path: str = "local_work"

    def __init__(self, **data):
        super().__init__(**data)
        args = ArgParser().args
        # TODO add override step
