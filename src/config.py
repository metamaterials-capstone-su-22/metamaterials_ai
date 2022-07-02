from pydantic import BaseModel


class Config(BaseModel):
    backward_batch_size: int = 2**7  # 2**9 512
    backward_lr: float = 1e-3  # tune.loguniform(1e-6, 1e-5),
    backward_num_epochs: int = 2500  # Default 2500
    create_plots = False
    data_file = "stainless-steel-revised-shuffled.pt"  # name of the data file
    data_folder: str = "local_data"  # Path to the data folder
    direction: str = "both"  # direct, inverse, both
    forward_batch_size: int = 2**7  # 2**9 512
    forward_lr: float = 1e-3  # tune.loguniform(1e-7, 1e-4),
    forward_num_epochs: int = 1600  # default 1600
    load_forward_checkpoint: bool = False
    load_backward_checkpoint: bool = False
    model_arch = 'ann'  # options 'MLPMixer', 'resnet1d' TODO: 'ann'
    num_gpu: int = 1  # number of GPU
    # TODO: Fix num_wavelens be set at load time
    num_wavelens: int | None = 800  # This will be set @ load time. ex. 800
    substrate: str = "stainless"  # options "stainless" , "inconel"
    # use_cache true means to use the .pt file instead of regenerating this
    use_cache: bool = True
    use_forward: bool = True
    should_verify_configs = True  # when true it does some config check before starting
    weight_decay = 1e-2  # default for AdamW 1e-2
    # Path to the working folder, checkpoint, graphs, ..
    work_folder: str = "local_work"

    def __init__(self, **data):
        super().__init__(**data)
        if(self.should_verify_configs):
            self.verify_config()

        # TODO add arg parser if needed
        # args = ArgParser().args

    def verify_config(self):
        '''Add constraints to configurations here to be checked'''
        if(not self.data_file.lower().startswith(self.substrate.lower())):
            print(
                f"Warning: Data file'{self.data_file}' does not match with substrate '{self.substrate}.")
            t = input(
                f"Do you want to continue?")
            if(not t == 'y'):
                raise Exception('Fix the mismatch and re-run!')
