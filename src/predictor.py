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
D_INC_CKPT_PATH = '../local_work/saved_best/D-1-res-ann-inconel-2022-07-15_23-44.ckpt' #../data/models/model.ckpt"
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
    
    def denormalize_and_decode(self, y_hat, substrate, change_dimension=False, normalize = "denormalize", round_laser_params = True):
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
        if normalize == 'normalize':
            y_final[:,0] = (y_hat[:,0] - min_speed) / (max_speed - min_speed)
            y_final[:,1] = (y_hat[:,1] - min_spacing)  /  (max_spacing - min_spacing)
        elif normalize == 'denormalize':
            y_final[:,0] = y_hat[:,0] * (max_speed - min_speed) + min_speed
            y_final[:,1] = y_hat[:,1]  * (max_spacing - min_spacing) + min_spacing
        else:
            y_final[:,0] = y_hat[:,0]
            y_final[:,1] = y_hat[:,1]

        if change_dimension:
            y_final[:,2] = torch.tensor([watts[i.item()] for i in watt_arg])
        else:
            y_final[:,2:] = y_hat[:,2:]

        if round_laser_params:
            y_final[:,0] = torch.round(y_final[:,0], decimals = -1)
            y_final[:,1] = torch.round(y_final[:,1], decimals = 0)
            
        return y_final
    def clean_and_send_to_lab(self, y_hat):
        """just reverse one hot encoding and get sorted unique parameters"""

        watts = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        watt_arg = torch.argmax(y_hat[:,2:], axis = 1)
        y_final = torch.empty((y_hat.shape[0], 3), dtype=torch.float32)
        y_final[:,0] = y_hat[:,0]
        y_final[:,1] = y_hat[:,1]
        y_final[:,2] = torch.tensor([watts[i.item()] for i in watt_arg])

        unique_data = sorted([list(x) for x in set(tuple(x) for x in y_final.tolist())], key = lambda x: (x[2], x[0], x[1]))

        stainless_send = []
        for i in unique_data:
            stainless_send.append([i[0], i[1], np.round(i[2], decimals = 1)])
        
        return stainless_send, len(stainless_send), len(y_final)

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

    def get_data(self, filepath, type):
        stain_data = torch.load(
                        Path(filepath))#["laser_params"] #stainless-steel-revised-shuffled inconel-revised-raw-shuffled
        wave_data = stain_data["interpolated_wavelength"]
        emiss_data = stain_data["interpolated_emissivity"]
        laser_params = stain_data["laser_params"]
        uids = stain_data["uids"]
        if type == 'test':
            wave_test = wave_data[round(len(wave_data) * .9):]  
            emiss_test = emiss_data[round(len(wave_data) * .9):]
            laser_params = laser_params[round(len(wave_data) * .9):]
            uids = uids[round(len(wave_data) * .9):]
        elif type == 'trainval':
            wave_test = wave_data[:round(len(wave_data) * .9)]  
            emiss_test = emiss_data[:round(len(wave_data) * .9)]
            laser_params = laser_params[:round(len(wave_data) * .9)]
            uids = uids[:round(len(wave_data) * .9)]
        else: 
            wave_test = wave_data
            emiss_test = emiss_data
            laser_params = laser_params
        
        return wave_test, emiss_test, laser_params, uids


    def find_in_data(self, y_hat, substrate):
        not_in_data = []
        same_params_uids = []
        pred_in_data = {"laser_params": [], "emissivity": [], "uids": []}
        test_in_data = {"laser_params": [], "emissivity": [], "uids": []}

        if substrate == 'steel':
            wave, emiss, laser_params, uids = self.get_data(SS_DATA_PATH, "full")
            wave_test, emiss_test, laser_params_test, uids_test = self.get_data(SS_DATA_PATH, "test")
        else:
            wave, emiss, laser_params, uids = self.get_data(INC_DATA_PATH, "full")
            wave_test, emiss_test, laser_params_test, uids_test = self.get_data(INC_DATA_PATH, "test")

        #[]#torch.tensor(type = "torch.float32")
        for i in range(len(y_hat)):
            val = torch.all(laser_params == y_hat[i], axis=1).nonzero().flatten()
            if len(val) == 0:
                not_in_data.append(y_hat[i])
            else:
                #if the predicted parameters = the original parameters
                same_params = torch.all(torch.eq(y_hat[i], laser_params_test[i])).item()
                if same_params:
                        same_params_uids.append(uids[val])
                #if the predicted != original parameters, add to two separate dictionaries
                else:
                    pred_in_data["laser_params"].append(laser_params[val].flatten())
                    pred_in_data["emissivity"].append(emiss[val].flatten())
                    pred_in_data["uids"].append(uids[val].flatten())
                    test_in_data["laser_params"].append(laser_params_test[i].flatten())
                    test_in_data["emissivity"].append(emiss_test[i].flatten())
                    test_in_data["uids"].append(uids_test[i].flatten())
        #list of tensors --> tensors
        if len(not_in_data) != 0:
            not_in_data = torch.stack(not_in_data)
        if len(test_in_data) != 0:
            test_in_data["laser_params"] = torch.stack(test_in_data["laser_params"])
            test_in_data["emissivity"] = torch.stack(test_in_data["emissivity"])
            test_in_data["uids"] = torch.stack(test_in_data["uids"])
        pred_in_data["laser_params"] = torch.stack(pred_in_data["laser_params"])
        pred_in_data["emissivity"] = torch.stack(pred_in_data["emissivity"])
        pred_in_data["uids"] = torch.stack(pred_in_data["uids"])

                
        return not_in_data, same_params_uids, pred_in_data, test_in_data


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
    
    def SAD(self, preds, test, substrate):
        """take in laser parameters and calculate sum of absolute difference."""
        #normalize parameters to calculate SAD
        pred_params = self.denormalize_and_decode(preds["laser_params"], substrate = substrate, change_dimension=True, normalize = "normalize", round_laser_params = False)
        original_params = self.denormalize_and_decode(test["laser_params"], substrate = substrate, change_dimension=True, normalize = "normalize", round_laser_params = False)

        #non normalized laser parameters for output
        pred = self.denormalize_and_decode(preds["laser_params"], substrate = substrate, change_dimension=True, normalize = "None", round_laser_params = True)
        original = self.denormalize_and_decode(test["laser_params"], substrate = substrate, change_dimension=True, normalize = "None", round_laser_params = True)
        #calculate SAD
        difference = torch.abs(torch.sub(pred_params, original_params))
        sad_values = torch.sum(difference, dim = 1)
        rmse_val = torch.tensor([rmse(preds["emissivity"][i], test["emissivity"][i]) for i in range(len(preds["emissivity"]))])

        return pred, original, sad_values.reshape(sad_values.shape[0], 1), rmse_val.reshape(rmse_val.shape[0], 1)


    @staticmethod
    def rmse_metrics(y_hat, y_pred, decimals = 5):

        rmse_scores = [rmse(y_hat[i], y_pred[i]) for i in range(len(y_hat))]
        return np.round(np.mean(rmse_scores), decimals = decimals), np.round(np.std(rmse_scores), decimals = decimals)