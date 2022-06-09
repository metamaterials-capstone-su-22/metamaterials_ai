from pydantic import BaseModel

from config import Config


class ModelConfig(BaseModel):
    arch: str
    direction: str
    num_classes: int | None
    in_channels: int | None
