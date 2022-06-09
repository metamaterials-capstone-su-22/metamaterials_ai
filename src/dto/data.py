from pydantic import BaseModel
from torch import FloatTensor, LongTensor


class Data(BaseModel):
    norm_laser_params: FloatTensor
    interp_emissivities: FloatTensor
    uids: LongTensor
    wavelength: FloatTensor

    class Config:
        arbitrary_types_allowed = True
