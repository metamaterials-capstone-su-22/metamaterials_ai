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
INC_DATA_PATH = '../local_data/inconel-onehot.pt'
SS_DATA_PATH = '../local_data/steel-onehot.pt'

# MODEL_PATH = "data/mymodels"

class Predictor:
    def __init__(
        self, 
        I_INC_CKPT_PATH, 
        I_SS_CKPT_PATH,  
        D_INC_CKPT_PATH, 
        D_SS_CKPT_PATH,
        round_laser_params = False,
            ) -> None:
        # Inputs: 
        # pass in emissivity curves into 'desired' 

        inc_params = torch.load(Path(INC_DATA_PATH))["laser_params"]
        steel_params = torch.load(Path(SS_DATA_PATH))["laser_params"]

        self.round_laser_params = round_laser_params

        self.I_INC_CKPT_PATH = I_INC_CKPT_PATH, 
        self.I_SS_CKPT_PATH =I_SS_CKPT_PATH,  
        self.D_INC_CKPT_PATH = D_INC_CKPT_PATH, 
        self.D_SS_CKPT_PATH = D_SS_CKPT_PATH,

        self.ss_max_speed, self.ss_max_spacing = steel_params.max(0)[0][0].item(), steel_params.max(0)[0][1].item()
        self.ss_min_speed, self.ss_min_spacing = steel_params.min(0)[0][0].item(), steel_params.min(0)[0][1].item()
        self.inc_max_speed, self.inc_max_spacing = inc_params.max(0)[0][0].item(), inc_params.max(0)[0][1].item()
        self.inc_min_speed, self.inc_min_spacing = inc_params.min(0)[0][0].item(), inc_params.min(0)[0][1].item()
    
    def denormalize_and_decode(self, y_hat, substrate, change_dimension=False):
        """input: y_hat.shape[0], 14 tensor
            output: y_hat.shape[0], 3 tensor with wattage no longer one hot encoded

            substrate:  pick the substrate, options: "inconel", other
            change_dimension: If you want to reverse the one-hot encoding applied to the wattage
                This changes output size from (shape[0],14) to (shape[0],3)"""
        if substrate == "inconel":
            max_speed = self.inc_max_speed
            min_speed = self.inc_min_speed
            max_spacing = self.inc_max_spacing
            min_spacing = self.inc_min_spacing
        else:
            max_speed = self.ss_max_speed
            min_speed = self.ss_min_speed
            max_spacing = self.ss_max_spacing
            min_spacing = self.ss_min_spacing

        watts = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        watt_arg = torch.argmax(y_hat[:,2:], axis = 1)
        dim = 14
        if change_dimension:
            dim = 3
        y_final = torch.empty((y_hat.shape[0], dim), dtype=torch.float32)

        y_final[:,0] = y_hat[:,0] * (max_speed - min_speed) + min_speed
        y_final[:,1] = y_hat[:,1]  * (max_spacing - min_spacing) + min_spacing
        if change_dimension:
            y_final[:,2] = torch.tensor([watts[i.item()] for i in watt_arg])
        else:
            y_final[:,2:] = y_hat[:,2:]

        if self.round_laser_params:
            y_final[:,0] = torch.round(y_final[:,0], decimals = -1)
            y_final[:,1] = torch.round(y_final[:,1], decimals = 0)
            
        return y_final

    def run_model(self, input, substrate, direction):
        #select correct path
        if substrate == "inconel" and direction == "inverse":
            path = Path(I_INC_CKPT_PATH)
        elif substrate == "inconel" and direction == "direct":
            path = Path(D_INC_CKPT_PATH)
        elif substrate == "steel" and direction == "direct":
            path = Path(D_SS_CKPT_PATH)
        elif substrate == "steel" and direction == "inverse":
            path = Path(I_SS_CKPT_PATH)
        else:
            raise Exception(f'Model path not correctly specified or does not exist at {path}!')

        #select direction of model
        if direction == "inverse":
            model = InverseModel.load_from_checkpoint(path, strict=False)
        elif direction == "direct":
            model = DirectModel.load_from_checkpoint(path, strict=False)
        else:
            raise Exception(f'Model direction not specified {direction}!')

        with torch.no_grad():
            y_hat = model(input)

        return y_hat

    def find_in_data(self, y_hat, substrate = 'steel'):
        not_in_data = []
        ss_in_data_same_params = []
        ss_in_data_diff_params = []
        if substrate == 'steel':
            params = self.steel_params
        else:
            params = self.inc_params
        indexes = []
        for i in range(len(y_hat)):
            val = torch.all(params == y_hat[i], axis=1).nonzero().flatten()
            if len(val) == 0:
                    indexes.append(-1)
            else:
                    indexes.append(val.numpy().item()) 
        return indexes, not_in_data


    def run_direct(self, y_hat_ss, y_hat_inc, laser_constraint  = False):

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
    
    def rmse_metrics(y_hat, y_pred):

        rmse_scores = [rmse(y_hat[i], y_pred[i]) for i in range(len(y_hat))]

        return np.mean(rmse_scores), np.stdev(rmse_scores)