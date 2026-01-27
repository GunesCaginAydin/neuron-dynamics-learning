import torch
import torch.nn as nn
import hydra

from hydra.core.hydra_config import HydraConfig
from baseinvertible import InvertibleProcess

class Diffusion(InvertibleProcess):
    def __init__(self):
        super.__init__()

    def train(self):
        pass

    def infer(self):
        pass
