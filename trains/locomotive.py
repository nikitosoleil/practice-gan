from abc import ABC
from typing import Dict

import torch
from apex import amp
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset

from configs import Config
from models import BaseModel
from utils.checkpointer import Checkpointer
from utils.reporter import Reporter
from .base_train import BaseTrain


class Locomotive(BaseTrain, ABC):
    def __init__(self, dataset: Dataset, models: Dict[str, BaseModel], optimizers: Dict[str, Optimizer],
                 reporter: Reporter, checkpointer: Checkpointer):
        self.models, self.optimizers, self.reporter, self.checkpointer = models, optimizers, reporter, checkpointer
        self.dataset = dataset

    def loose(self, loss, model, phase, batch):
        if Config.opt_level is not None:
            with amp.scale_loss(loss / Config.accumulation_steps[phase],
                                self.optimizers[model]) as scaled_loss:
                scaled_loss.backward()
        else:
            (loss / Config.accumulation_steps[phase]).backward()

        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizers[model]),
                                       Config.max_norm[phase])

        if (batch + 1) % Config.accumulation_steps[phase] == 0:
            if Config.is_stepping[phase]:
                self.optimizers[model].step()
            self.optimizers[model].zero_grad()
