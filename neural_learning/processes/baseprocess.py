import torch
import torch.nn as nn
import hydra

from hydra.core.hydra_config import HydraConfig

class Process():
    def __init__(self, config):
        self._config = config

    @property
    def config_settings(self):
        return self._config

    def train(self):
        raise NotImplementedError

    def infer(self):
        raise NotImplementedError
    
    def save(self):
        pass
