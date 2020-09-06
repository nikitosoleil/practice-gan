from abc import ABC, abstractmethod


class Averager(ABC):
    def __init__(self, initial_value=0):
        self.initial_value = initial_value

    @abstractmethod
    def upd(self, val):
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class MeanAverager(Averager):
    """
    Arithmetic mean
    """

    def __init__(self, initial_value=0):
        super().__init__(initial_value)
        self.sum, self.cnt = None, None
        self.reset()

    def upd(self, val):
        if not isinstance(val, dict):
            val = {'__default': val}
        for k, v in val.items():
            self.sum[k] = self.sum.get(k, self.initial_value) + v
        self.cnt += 1

    def get(self):
        if self.cnt == 0:
            return None
        else:
            res = {k: v / self.cnt for k, v in self.sum.items()}
            if list(res.keys()) == ['__default']:
                return res['__default']
            else:
                return res

    def reset(self):
        self.sum = dict()
        self.cnt = 0
