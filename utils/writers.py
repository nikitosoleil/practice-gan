import logging
from abc import ABC, abstractmethod
from typing import List

from torch.utils.tensorboard import SummaryWriter


class Writer(ABC):
    @abstractmethod
    def write(self, name, value, time):
        pass


class TBWriter(Writer):
    def __init__(self, tb_writer: SummaryWriter):
        self.tb_writer = tb_writer

    def write(self, name, value, time):
        if isinstance(value, str):
            self.tb_writer.add_text(name, value, time)
        elif isinstance(value, dict):
            self.tb_writer.add_scalars(name, value, time)
        else:
            self.tb_writer.add_scalar(name, value, time)


class TerminalWriter(Writer):
    def write(self, name, value, time):
        logging.info(f'{name} {value}')


class CombinedWriter(Writer):
    def __init__(self, writers: List[Writer]):
        self.writers = writers

    def write(self, name, value, time):
        for writer in self.writers:
            writer.write(name, value, time)
