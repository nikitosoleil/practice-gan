from typing import Dict

from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from configs import Config
from models import BaseModel
from trains import BaseTrain


class GANInteract(BaseTrain):
    def __init__(self, models: Dict[str, BaseModel]):
        self.generator = models['generator']

    def choochoo(self):
        while True:
            z = torch.randn((Config.val_samples, Config.latent_dim))
            gen_imgs = self.generator.train(z)[0]
            gen_imgs = gen_imgs.detach().cpu()
            img = make_grid(gen_imgs, normalize=True)
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
            plt.show()
