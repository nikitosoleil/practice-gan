import os

import torch
from apex import amp

from configs import Config


class Checkpointer:
    """
    Helper class for handling model checkpointing
    """

    def __init__(self, path):
        self.path = path
        os.makedirs(self.path)

    def save(self, models, optimizers, time: int):
        """
        Save given models and optimizers on given time

        :param models: models to save
        :param optimizers: optimizers state dicts of which to save
        :param time: given time
        """

        checkpoint_path = os.path.join(self.path, f'time_{time}')

        for name in Config.trainable_models:
            models[name].save(checkpoint_path)

        optimizer_state_dicts = {name: optimizer.state_dict() for name, optimizer in optimizers.items()}
        torch.save(optimizer_state_dicts, os.path.join(checkpoint_path, 'optimizers.pth'))

        if Config.opt_level is not None:
            torch.save(amp.state_dict(), os.path.join(checkpoint_path, 'amp.pth'))
