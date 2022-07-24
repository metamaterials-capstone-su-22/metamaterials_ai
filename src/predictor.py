from config import Config
from models.base_model import BaseModel
from models.direct_model import DirectModel
from models.inverse_model import InverseModel
from models.model_factory import ModelFactory
from pathlib import Path
from src.utils import FileUtils
import torch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from utils import rmse
import os

config = Config()
I_INC_CKPT_PATH = '../local_work/saved_best/I-1-res-ann-inconel.ckpt' #../data/models/model.ckpt"
I_SS_CKPT_PATH = '../local_work/saved_best/I-1-res-ann-stainless.ckpt' #../data/models/model.ckpt"
D_INC_CKPT_PATH = '../local_work/saved_best/D-1-res-ann-inconel.ckpt' #../data/models/model.ckpt"
D_SS_CKPT_PATH = '../local_work/saved_best/D-1-res-ann-stainless.ckpt' #../data/models/model.ckpt"
INC_DATA_PATH = '../local_data/inconel-revised-shuffled.pt'
SS_DATA_PATH = '../local_data/stainless-steel-revised-shuffled.pt'

# MODEL_PATH = "data/mymodels"

class Predictor:
    def __init__(self, desired, include_inconel = True, include_stainless = True, round_laser_params = True) -> None:
        self.desired = desired
        self.include_inconel = include_inconel
        self.include_stainless = include_stainless
        inc_params = torch.load(Path(INC_DATA_PATH))["laser_params"]
        steel_params = torch.load(Path(SS_DATA_PATH))["laser_params"]
        self.round_laser_params = round_laser_params
        self.ss_max_speed, self.ss_max_spacing = steel_params.max(0)[0][0].item(), steel_params.max(0)[0][1].item()
        self.ss_min_speed, self.ss_min_spacing = steel_params.min(0)[0][0].item(), steel_params.min(0)[0][1].item()
        self.inc_max_speed, self.inc_max_spacing = inc_params.max(0)[0][0].item(), inc_params.max(0)[0][1].item()
        self.inc_min_speed, self.inc_min_spacing = inc_params.min(0)[0][0].item(), inc_params.min(0)[0][1].item()
    
    def denormalize_decode_result(y_hat, max_speed, max_spacing, min_speed, min_spacing):
        """input: 1,14 tensor
            output: 1,3 tensor with wattage no longer one hot encoded"""
        watts = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

        watt_arg = torch.argmax(y_hat[0][2:])
        y_final = torch.empty(3, dtype=torch.float32)
        y_final[0] = max_speed * y_hat[0][0] # TODO call the scale
        y_final[1] = max_spacing * y_hat[0][1] # TODO call the scale
        y_final[2]= watts[watt_arg]
        return y_final

    def run_inverse(self, input):
        x = input
        #run INC Inverse        
        print(os.getcwd())
        INC_PATH = Path(I_INC_CKPT_PATH)
        if not Path.is_file(INC_PATH):
            raise Exception(f'Model file does not exist at {INC_PATH}!')
        i_model_ss = InverseModel.load_from_checkpoint(INC_PATH, strict=False)
        i_model_ss.eval()
        with torch.no_grad():
            y_hat_ss = i_model_ss(x)

        #run SS Inverse
        SS_PATH = Path(I_SS_CKPT_PATH)
        if not Path.is_file(SS_PATH):
            raise Exception(f'Model file does not exist at {SS_PATH}!')
        i_model_inc = InverseModel.load_from_checkpoint(SS_PATH, strict=False)
        i_model_inc.eval()
        with torch.no_grad():
            y_hat_inc = i_model_inc(x)

        #return ss and inc laser parameter preds
        return y_hat_ss, y_hat_inc

    def run_direct(self, y_hat_ss, y_hat_inc):
     

        #run direct inc model
        direct_inc_filepath = Path(D_INC_CKPT_PATH) #CHANGEME
        if not Path.is_file(direct_inc_filepath):
            raise Exception(f'Model file does not exist at {direct_inc_filepath}!')
        d_inc_model = DirectModel.load_from_checkpoint(direct_inc_filepath)
        d_inc_model.eval()
        with torch.no_grad():
            direct_inc_emiss_y_hat = d_inc_model(y_hat_inc.reshape(1,14))
        
        #run inverse inc model
        direct_ss_filepath = Path(D_INC_CKPT_PATH) #CHANGEME
        if not Path.is_file(direct_ss_filepath):
            raise Exception(f'Model file does not exist at {direct_ss_filepath}!')
        d_ss_model = DirectModel.load_from_checkpoint(direct_ss_filepath)
        d_ss_model.eval()
        with torch.no_grad():
            direct_ss_emiss_y_hat = d_ss_model(y_hat_ss.reshape(1,14))
        
        #retrn 2 emiss predictions
        return direct_ss_emiss_y_hat, direct_inc_emiss_y_hat

    def denormalize_decode_result(self, y_hat, max_speed, max_spacing, min_speed, min_spacing):
        """input: 1,14 tensor
            output: 1,3 tensor with wattage no longer one hot encoded"""
        watts = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

        watt_arg = torch.argmax(y_hat[0][2:])
        y_final = torch.empty(3, dtype=torch.float32)
        y_final[0] = ((max_speed - min_speed) * y_hat[0][0]) + min_speed # TODO call the scale
        y_final[1] = ((max_spacing-min_spacing) * y_hat[0][1]) + min_spacing # TODO call the scale
        y_final[2]= watts[watt_arg]
        return y_final

    def best_predictor(self, include_inconel, include_stainless, direct_ss_emiss_y_hat, direct_inc_emiss_y_hat, desired_emiss):
        substrates = ['inconel', 'stainless']
        best_params = []
        best_rmse = 0
        if not include_inconel and not include_stainless:
            raise Exception(f'Neither inconel or stainless was selected!!')
        if (include_inconel and include_stainless and rmse(direct_inc_emiss_y_hat, desired_emiss) < rmse(direct_ss_emiss_y_hat, desired_emiss)) or (include_inconel and not include_stainless):
            best_substrate = substrates[0]
            best_params = self.denormalize_decode_result(direct_inc_emiss_y_hat, self.inc_max_speed, self.inc_max_spacing, self.inc_min_speed, self.inc_min_spacing)
            best_rmse = rmse(direct_inc_emiss_y_hat, desired_emiss).item()
        else:
            best_substrate = substrates[1]
            best_params = self.denormalize_decode_result(direct_ss_emiss_y_hat, self.ss_max_speed, self.ss_max_spacing, self.ss_min_speed, self.ss_min_spacing)
            best_rmse = rmse(direct_ss_emiss_y_hat, desired_emiss).item()
        
        best_params = [torch.round(best_params[0], decimals = 1).item(), torch.round(best_params[1], decimals = 1).item(), torch.round(best_params[2], decimals = 1).item()]
        return best_substrate, best_params, best_rmse