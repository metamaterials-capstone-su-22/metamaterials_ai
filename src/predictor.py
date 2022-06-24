
from models.forward_model import ForwardModel
from models.backward_model import BackwardModel
from models.base_model import BaseModel
from config import Config
from models.model_factory import ModelFactory

CKPT_PATH = "../data/models/model.ckpt"
MODEL_PATH = "data/mymodels"

config = Config()

model = ModelFactory(config).create_model('forward')

model = model.load_from_checkpoint(CKPT_PATH)

ckpt_path = CKPT_PATH

meta_trainer = MetaTrainerFactory(config, gpus=0)
forward_model = None
if config.use_forward:
    forward_trainer = meta_trainer.create_meta_trainer("forward")
    if not config.load_forward_checkpoint:
        forward_trainer.fit()
    forward_trainer.test()
    forward_model = forward_trainer.model

# Close the Forward before backward if you want separate project
wandb.finish()
train_backward(meta_trainer, forward_model)



prediction = model.trainer.predict(
    model=model,
    ckpt_path=ckpt_path,
    datamodule=None,
    return_predictions=True,
)
