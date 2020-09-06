import torch

from .model import BaseModel


class WGANDiscriminator(BaseModel):
    def metric_run(self, *args, **kwargs):
        with torch.enable_grad():
            loss, metrics = self.forward(*args, **kwargs)[:2]
        return loss, metrics
