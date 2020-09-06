from typing import Iterable, Iterator


class PerpetualLoader(Iterator):
    def __init__(self, loader: Iterable):
        self.loader = loader
        self.iter = iter(self.loader)

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return next(self)
