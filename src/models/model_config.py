from pydantic import BaseModel


class ModelConfig(BaseModel):
    arch: str
    direction: str
    num_classes: int | None
    in_channels: int | None
