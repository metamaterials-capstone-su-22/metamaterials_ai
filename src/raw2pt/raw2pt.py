#!/usr/bin/env python3
# TODO: (PM) needs clean up
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pymongo
import pytorch_lightning as pl
import sklearn
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tqdm.contrib import tenumerate

LaserParams, Emiss = torch.FloatTensor, torch.FloatTensor


def parse_entry(filename: Path, db: list) -> None:
    pattern = re.compile(
        pattern=re.compile(r"Power_(\d)_(\d)_W_Speed_(\d+)_mm_s_Spacing_(\d+)_um.txt")
        # r"Power_(\d)_(\d)_W_Speed_(.+)_mm_s_Spacing_(.+)_um\s*(\d+)\s*\.txt"
    )
    m = pattern.match(filename.name)
    if m is None:
        return
        # raise ValueError(f"Could not parse filename {filename}")
    power, x_speed, y_spacing = (
        int(m[1]) + int(m[2]) * 10**-1,
        float(m[3]),
        float(m[4]),
    )

    raw_data = pd.read_csv(
        filename, header=None, names=["wavelens", "emisses"], delim_whitespace=True
    )

    wavelens = raw_data.wavelens
    emisses = raw_data.emisses

    entry = {
        "laser_repetition_rate_kHz": 100,
        "laser_wavelength_nm": 1030,
        "laser_polarization": "p-pol",
        "laser_steering_equipment": "Galvano scanner",
        "laser_hardware_model": "s-Pulse (Amplitude)",
        "substrate_details": "SS foil with 0.5 mm thickness (GF90624180-20EA)",
        "laser_power_W": power,
        "laser_scanning_speed_x_dir_mm_per_s": x_speed,
        "laser_scanning_line_spacing_y_dir_micron": y_spacing,
        "substrate_material": "stainless_steel",
        "emissivity_spectrum": [
            {"wavelength_micron": w, "normal_emissivity": e}
            for w, e in zip(wavelens, emisses)
        ],
        "emissivity_averaged_over_frequency": sum(emisses) / len(emisses),
    }
    db.append(entry)


def raw_to_pt(
    use_cache: bool = True,
    num_wavelens: int = 800,
    db: list = [],
    output_path="data/pt/data.pt",
) -> Tuple[LaserParams, Emiss, torch.LongTensor]:
    """Data is sorted in ascending order of wavelength."""
    if all(
        [
            use_cache,
            Path("data/pt/data.pt").exists(),
        ]
    ):
        data = torch.load(Path("/data/pt/data.pt"))
        norm_laser_params, interp_emissivities, uids = (
            data["normalized_laser_params"],
            data["interpolated_emissivity"],
            data["uids"],
        )

        # XXX check length to avoid bugs.
        if interp_emissivities.shape[-1] == num_wavelens:
            return norm_laser_params, interp_emissivities, uids

    laser_params, emissivity, wavelength = [], [], []
    interp_emissivities, interp_wavelengths = [], []
    uids = []
    # TODO: clean up and generalize when needed
    # the values are indexes for one hot vectorization
    wattage_idxs = {
        0.2: 0,
        0.3: 1,
        0.4: 2,
        0.5: 3,
        0.6: 4,
        0.7: 5,
        0.8: 6,
        0.9: 7,
        1.0: 8,
        1.1: 9,
        1.2: 10,
        1.3: 11,
        # these last 2 wattages are problematic since their
        # emissivities are different lengths
        # 1.4: 12,
        # 1.5: 13,
    }

    for uid, entry in tenumerate(db):
        emiss_plot: List[float] = [
            e
            for ex in entry["emissivity_spectrum"]
            if ((e := ex["normal_emissivity"]) != 1.0 and ex["wavelength_micron"] < 12)
        ]
        wavelength_plot: List[float] = [
            ex["wavelength_micron"]
            for ex in entry["emissivity_spectrum"]
            if (ex["normal_emissivity"] != 1.0 and ex["wavelength_micron"] < 12)
        ]
        # Reverse to sort in ascending rather than descending order.
        emiss_plot.reverse()
        wavelength_plot.reverse()

        # Interpolate to `num_wavelens` points.
        interp_wavelen = np.linspace(
            min(wavelength_plot), max(wavelength_plot), num=num_wavelens
        )
        interp_emiss = interp1d(wavelength_plot, emiss_plot)(interp_wavelen)

        # drop all problematic emissivity (only 3% of data dropped)

        if len(emiss_plot) != (_MANUALLY_COUNTED_LENGTH := 821) or any(
            not (0 <= x <= 1) for x in emiss_plot
        ):
            continue

        # if entry["laser_power_W"] > 1.3:
        #     print(entry["laser_power_W"])

        params = [
            entry["laser_scanning_speed_x_dir_mm_per_s"],
            entry["laser_scanning_line_spacing_y_dir_micron"],
            # wattages are converted to one hot indexes
            *F.one_hot(
                torch.tensor(wattage_idxs[round(entry["laser_power_W"], 1)]),
                num_classes=len(wattage_idxs),
            ),
        ]

        uids.append(uid)
        laser_params.append(params)
        emissivity.append(emiss_plot)
        wavelength.append(wavelength_plot)
        interp_emissivities.append(interp_emiss)
        interp_wavelengths.append(interp_wavelen)

    # normalize laser parameters
    laser_params = torch.FloatTensor(laser_params)
    emissivity = torch.FloatTensor(emissivity)
    wavelength = torch.FloatTensor(wavelength)
    interp_emissivities = torch.FloatTensor(interp_emissivities)
    interp_wavelengths = torch.FloatTensor(interp_wavelengths)
    uids = torch.LongTensor(uids)

    print(f"{len(laser_params)=}")
    print(f"{len(emissivity)=}")
    print(f"{laser_params.min(0)=}")
    print(f"{laser_params.max(0)=}")
    print(f"{emissivity.min()=}")
    print(f"{emissivity.max()=}")

    # Save unnormalized data for convenience later.

    norm_laser_params = laser_params / laser_params.max(0).values
    torch.save(
        {
            "wavelength": wavelength,
            "laser_params": laser_params,
            "emissivity": emissivity,
            "uids": uids,
            "interpolated_emissivity": interp_emissivities,
            "interpolated_wavelength": interp_wavelengths,
            "normalized_laser_params": norm_laser_params,
        },
        Path(output_path),
    )

    return norm_laser_params, interp_emissivities, uids


print("Process started")
input_path = "data/raw/stainless-steel-revised"
output_pt = "data/pt/stainless-steel-revised.pt"
data_emiss = []
for p in Path(input_path).rglob("*.txt"):
    # print(p)
    parse_entry(p, data_emiss)

print(len(data_emiss))

laser_params, emiss, uids = raw_to_pt(
    use_cache=False, num_wavelens=800, db=data_emiss, output_path=output_pt
)

print("Done!")
