from config import Config
from models.backward_model import InverseModel
from models.base_model import BaseModel
from models.forward_model import DirectModel
from models.model_factory import ModelFactory

CKPT_PATH = "../data/models/model.ckpt"
MODEL_PATH = "data/mymodels"

config = Config()

model = ModelFactory(config).create_model("direct")

model = model.load_from_checkpoint(CKPT_PATH)

ckpt_path = CKPT_PATH

meta_trainer = MetaTrainerFactory(config, gpus=0)
driect_model = None
if config.use_direct:
    direct_trainer = meta_trainer.create_meta_trainer("direct")
    if not config.load_direct_checkpoint:
        direct_trainer.fit()
    direct_trainer.test()
    driect_model = direct_trainer.model

# Close the Direct before Inverse if you want separate project
wandb.finish()
train_inverse(meta_trainer, driect_model)


prediction = model.trainer.predict(
    model=model,
    ckpt_path=ckpt_path,
    datamodule=None,
    return_predictions=True,
)
