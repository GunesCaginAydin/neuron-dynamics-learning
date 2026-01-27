from typing import Optional

import torch
import torch.nn as nn

def RMSE(pred: torch.Tensor, act: torch.Tensor, reduction: Optional[str]):
    assert isinstance(pred,torch.Tensor) and isinstance(act,torch.Tensor)
    assert pred.device == act.device
    mse = 0
    rmse = 0
    return rmse

def R2(pred: torch.Tensor, act: torch.Tensor, reduction: Optional[str]):
    assert isinstance(pred,torch.Tensor) and isinstance(act,torch.Tensor)
    assert pred.device == act.device
    mse = 0
    rmse = 0
    return rmse