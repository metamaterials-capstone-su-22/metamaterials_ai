from pydantic import BaseModel
from torch import FloatTensor, LongTensor


class Data(BaseModel):
    laser_params: FloatTensor
    emiss: FloatTensor
    uids: LongTensor
    wavelength: FloatTensor

    class Config:
        arbitrary_types_allowed = True
