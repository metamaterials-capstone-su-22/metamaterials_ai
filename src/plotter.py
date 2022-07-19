import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import utils
from config import Config


class Plotter:
    @staticmethod
    def plot_results(preds, direct_model, inverse_model, config: Config):
        work_folder = config.work_folder
        # graph(residualsflag = True, predsvstrueflag = True, index_str = params_str, target_str = save_str)
        true_emiss = preds[0]["true_emiss"]
        pred_array = []

        # Variant num is the number of random curves to generate with jitter
        variant_num = 1
        # Arbitrary list is the indices you want to look at in a tensor of emissivity curves. In the FoMM case, 0 = cutoff at 2.5 wl, 800 = cutoff at 12.5 wl.
        arbitrary_list = [50, 51, 52]

        print("start plotting ...")
        for i in range(variant_num):
            # new_true = [torch.tensor(emiss+random.uniform(-0.05, 0.05)) for emiss in true_emiss]
            # jitter isn't doing anything XXX
            random_mult = random.uniform(-0.3, 0.3)
            new_true = torch.clamp(
                torch.tensor(
                    [
                        [
                            (random_mult * (1 / emiss) * (e_index / 3 + 100) / 600)
                            * emiss
                            + emiss
                            for e_index, emiss in enumerate(sub_emiss)
                        ]
                        for sub_emiss in true_emiss
                    ]
                ),
                0,
                1,
            )

            if i == 0:
                new_true = true_emiss
            back = inverse_model(new_true)
            # add spacing

            # minspeed = 10, maxspeed = 700

            # min 1 max 42

            new_pred = direct_model(back)
            pred_array.append(new_pred.detach())

        for i in arbitrary_list:
            pred_emiss = []
            for j in range(variant_num):
                pred_emiss.append(pred_array[j][i])
            pred_emiss = torch.stack(pred_emiss)
            fig = Plotter.plot_val(pred_emiss, true_emiss[i], i, config)
            fig.savefig(
                f"{work_folder}/figs/{i}_predicted_{utils.get_formatted_utc()}.png",
                dpi=300,
            )
            plt.close(fig)

    @staticmethod
    def plot_val(pred_emiss, true_emiss, index, config):
        data_folder = config.data_folder
        wavelen = torch.load(f"{data_folder}/{config.data_file}")[
            "interpolated_wavelength"
        ][0]
        pred_emiss = pred_emiss[0]
        extended_max = 2.5
        extended_min = 0.1

        granularity = 192

        extension = torch.tensor(
            [
                extended_min + (i) / granularity * (extended_max - extended_min)
                for i in range(granularity)
            ]
        )

        extended_wave = torch.cat((extension, wavelen))

        # extend the pred emiss
        old_emiss = pred_emiss
        first_emiss = np.float(old_emiss[0])
        new_emiss = torch.cat(
            (torch.tensor([first_emiss for i in range(granularity)]), old_emiss)
        )
        pred_emiss = new_emiss

        # extend the true emiss
        old_emiss = true_emiss
        first_emiss = np.float(old_emiss[0])
        new_emiss = torch.cat(
            (torch.tensor([first_emiss for i in range(granularity)]), old_emiss)
        )
        true_emiss = new_emiss

        wavelen = extended_wave

        fig, ax = plt.subplots()
        temp = 1400
        planck = [float(utils.planck_norm(wavelength, temp)) for wavelength in wavelen]

        planck_max = max(planck)
        planck = [wave / planck_max for wave in planck]

        wavelen_cutoff = float(wavelen[index + granularity])
        # format the predicted params
        FoMM = utils.planck_emiss_prod(wavelen, pred_emiss, wavelen_cutoff, 1400)

        ax.plot(
            wavelen,
            pred_emiss,
            c="blue",
            alpha=0.2,
            linewidth=1.0,
            label=f"Predicted Emissivity, FoMM = {FoMM}",
        )
        ax.plot(
            wavelen,
            true_emiss,
            c="black",
            label=f"Ideal target emissivity",
            linewidth=2.0,
        )
        ax.legend()
        return fig
