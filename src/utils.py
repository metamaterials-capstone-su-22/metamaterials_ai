
from __future__ import annotations

from math import floor
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, TypedDict

import numpy as np
import torch
from torch import Tensor

Stage = Literal["train", "val", "test"]


def rmse(pred: Tensor, target: Tensor, epsilon=1e-8):
    """Root mean squared error.

    Epsilon is to avoid NaN gradients. See https://discuss.pytorch.org/t/rmse-loss-function/16540/3.
    """
    return (torch.nn.functional.mse_loss(pred, target) + epsilon).sqrt()
