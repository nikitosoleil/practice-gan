import torch

from configs import Status
from utils.averagers import MeanAverager
from utils.writers import Writer


class Reporter:
    def __init__(self, writer: Writer):
        self.writer = writer
        self.averagers = {}
        self.sep = '/'

    def __upd(self, s, value):
        if s not in self.averagers:
            self.averagers[s] = MeanAverager()
        self.averagers[s].upd(value)

    def report(self, mode, entity, name, value):
        s = self.sep.join([entity, mode, name])

        def __detensorify(x):
            return x.item() if isinstance(x, torch.Tensor) else x

        if isinstance(value, dict):
            value = {k: __detensorify(v) for k, v in value.items()}
        else:
            value = __detensorify(value)

        self.__upd(s, value)

    def full_report(self, mode, entity, loss, stats):
        self.report(mode, entity, 'loss', loss)
        for stat, value in stats.items():
            self.report(mode, entity, stat, value)

    def direct_report(self, name, value):
        self.writer.write(name, value, Status.time)

    def push(self, mode):
        for s, averager in self.averagers.items():
            if s.split(self.sep)[1] == mode:
                average = averager.get()
                self.writer.write(s, average, Status.time)
                averager.reset()
