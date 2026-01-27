from typing import Union, Optional, Tuple
import logging

from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class config():
    width: int = 128
    depth: int = 3
    inputdim : int = 2
    outputdim : int = 2
    variables: str = "v"
    type: str = "single"

class Block(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.block = nn.Sequential(
            nn.Conv1d(),
            nn.AvgPool1d()
        )
    def forward(self, x):
        x = self.block(x)
        return x
    
class NeuronBaseCNN(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.neuronbasecnn = nn.ModuleDict(
            b = [Block for _ in config.layers],
            pr = nn.Linear()
        )
    
    def forward(self, x):
        for block in self.b:
            x = block(x)
        x = self.pr(x)
        return x

