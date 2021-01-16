"""learner module"""

import os
import sys
import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm


class Trainer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_loader = None
        self.validation_loader = None
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = None
        self.current_train_batch = None
        self.current_valid_batch = None
        self._model_state = None
        self._train_state = None
        self._callback_handler = None
        self.device = None
        self.fp16 = None
        self.scaler = None
        self.metrics = {}
        self.metrics['train'] = {}
        self.metrics['valid'] = {}
        self.metrics['test'] = {}
    
    @property
    def model_state(self):
        return self._model_state
    @model_state.setter
    def model_state(self, value):
        self._model_state = value
    @property
    def train_state(self):
        return self._train_state
    @train_state.setter
    def train_state(self, value):
        self._train_state = value
        if self._callback_handler:
            self._callback_handler(value)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _init_model(self):
        pass
    def train_one_batch(self):
        pass
    def train_one_epoch(self):
        pass
    def validate_one_batch(self):
        pass
    def validate_one_epoch(self):
        pass
    def save(self):
        pass
    def load(self):
        pass
    def train(self):
        pass
    def validate(self):
        pass
    def predict(self):
        pass


if __name__ == '__main__':
    # define model
    # define datasets
    # initialize trainer
    # start training
    # validate the model performance
    # plot outputs
    # test
    pass
    
