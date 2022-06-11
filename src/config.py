from pydantic import BaseModel


class Config(BaseModel):
    backward_batch_size: int = 1000  # 2**9
    backward_lr: float = 1e-6  # tune.loguniform(1e-6, 1e-5),
    backward_num_epochs: int = 2  # 2500
    create_plots = False
    data_file = "stainless_steel.pt"  # name of the data file
    data_path: str = "local_data"  # Path to the data folder
    forward_batch_size: int = 1000  # 2**9
    forward_lr: float = 1e-6  # tune.loguniform(1e-7, 1e-4),
    forward_num_epochs: int = 2  # 1600
    load_forward_checkpoint: bool = True
    load_backward_checkpoint: bool = False
    model_arch = "MLPMixer"  # options 'MLPMixer'
    num_gpu: int = 1  # number of GPU
    # TODO: Fix num_wavelens be set at load time
    num_wavelens: int | None = 800  # This will be set @ load time. ex. 800
    substrate: str = "stainless_steel"  # options "stainless_steel" , "inconel"
    # use_cache truw means to use the .pt file instead of regeneratign this
    use_cache: bool = True
    use_forward: bool = True
    # Path to the working folder, checkpoint, graphs, ..
    work_path: str = "local_work"

    def __init__(self, **data):
        super().__init__(**data)
        # TODO add arg parser if needed
        # args = ArgParser().args
