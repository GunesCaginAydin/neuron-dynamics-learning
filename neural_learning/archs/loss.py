from typing import Optional

import torch
import torch.nn as nn

def MSE(pred: torch.Tensor, act: torch.Tensor, reduction: Optional[str]):
    assert isinstance(pred,torch.Tensor) and isinstance(act,torch.Tensor)
    assert pred.device == act.device
    return nn.MSELoss(pred,act,reduce=reduction)