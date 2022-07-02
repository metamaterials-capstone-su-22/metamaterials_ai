#!/usr/bin/env python3


from __future__ import annotations

import os
from math import floor
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, TypedDict

import numpy as np
import torch
from torch import Tensor
from datetime import datetime


Stage = Literal["train", "val", "test"]

# TODO replace with scanl


def split(n: int, splits: Optional[Mapping[Stage, float]] = None) -> Dict[Stage, range]:
    """
    n: length of dataset
    splits: map where values should sum to 1 like in `{"train": 0.8, "val": 0.1, "test": 0.1}`
    """
    if splits is None:
        splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    return {
        "train": range(0, floor(n * splits["train"])),
        "val": range(
            floor(n * splits["train"]),
            floor(n * splits["train"]) + floor(n * splits["val"]),
        ),
        "test": range(floor(n * splits["train"]) + floor(n * splits["val"]), n),
    }


def rmse(pred: Tensor, target: Tensor, epsilon=1e-8):
    """Root mean squared error.

    Epsilon is to avoid NaN gradients. See https://discuss.pytorch.org/t/rmse-loss-function/16540/3.
    """
    return (torch.nn.functional.mse_loss(pred, target) + epsilon).sqrt()


def step_tensor():
    """Returns tensor of wavelengths, sorted high to low."""
    # index at 0 because each row has the same info.
    wavelens = torch.load(
        Path("local_data/stainless_steel.pt"))["wavelength"][0]
    out = torch.zeros(len(wavelens), len(wavelens))
    for r, _ in enumerate(out):
        out[r, : r + 1] = 1.0
    return out


def planck_norm(wavelen, temp):
    e = 2.7182818
    c1 = 1.191042 * 10**8
    denom1 = e ** (1.4387752 * 10**4 / wavelen / temp) - 1
    denom2 = wavelen**5 * (denom1)
    emiss = c1 / denom2
    return emiss


def planck_emiss_prod(wave_x, emiss_y, wavelen, temp):
    """
    wave_x: x values, wavelength in um

    emiss_y: y values, emissivity

    wavelen: target wavelength for cutoff"""

    total_before = 0
    total_after = 0

    for i in range(len(wave_x) - 1):
        if wave_x[i] < wavelen:
            width = abs(wave_x[i + 1] - wave_x[i])
            value = planck_norm(wavelen, temp)
            total_before += width * value * emiss_y[i]
        else:
            width = abs(wave_x[i + 1] - wave_x[i])
            value = planck_norm(wavelen, temp)
            total_after += width * value * emiss_y[i]
    figure_of_merit_metric = total_before / total_after
    return figure_of_merit_metric


def step_at_n(n: float = 3.5, max: float = 12):
    """Returns tuple of x and y which corresponds to a function that outputs 0 at index < n, and 1 at index > n.
    n: wavelength to step down at
    max: max wavelength


    """

    wavelength = np.array(
        torch.load("/data/alok/laser/data.pt")["interpolated_wavelength"][0]
    )
    low = 0
    high = 0
    for i, wl in enumerate(wavelength):
        if wl < n:
            low = i
        if wl < max:
            high = i

    x = wavelength[low:high]
    y = [1 for i in range(low)] + [0 for i in range(high - low)]
    return (x, y)


def get_latest_chk_point_path(work_folder, direction):
    path = Path(f"{work_folder}/weights/{direction}")
    try:
        return str(
            max(
                path.glob("*.ckpt"),
                key=os.path.getctime,
            )
        )
    except Exception as e:
        print(
            f"Error: Could not load the latest check point for *.ckpt files! Check path {path}. Error. {e}"
        )
        print(
            f"Note: To start from scratch set the config flag to not load chekcpoint."
        )
        raise


def get_formatted_utc():
    return f'{datetime.utcnow().strftime("%Y-%m-%d_%H-%M")}'


def get_dated_postfix(meta_trainer):
    direction = meta_trainer.model.direction
    arch = meta_trainer.model.model_config.arch
    substrate = meta_trainer.config.substrate
    return f'{direction.title()[0]}-{arch}-{substrate}-{get_formatted_utc()}'

# def get_best_chk_point_path(work_folder, direction):
#     path = Path(f"{work_folder}/weights/{direction}")
#     try:
#         return str(
#             max(
#                 path.glob("*.ckpt"),
#                 key=os.path.getctime,
#             )
#         )
#     except Exception as e:
#         print(
#             f"Error: Could not load the latest check point for *.ckpt files! Check path {path}. Error. {e}"
#         )
#         print(
#             f"Note: To start from scratch set the config flag to not load chekcpoint."
#         )
#         raise
