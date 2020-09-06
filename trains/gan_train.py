import logging
from typing import Dict, Iterable, Iterator

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import trange

from configs import Config, Status
from iterators import PerpetualLoader
from models import BaseModel
from utils.checkpointer import Checkpointer
from utils.misc import train_test_split
from utils.reporter import Reporter
from utils.scheduler import Scheduler
from .locomotive import Locomotive


class WGANTrain(Locomotive):
    def __init__(self, dataset: Dataset, models: Dict[str, BaseModel], optimizers: Dict[str, Optimizer],
                 reporter: Reporter, checkpointer: Checkpointer):
        super().__init__(dataset, models, optimizers, reporter, checkpointer)

        self.dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                Config.data_path,
                train=True,
                download=True,
                transform=transforms.Compose([transforms.Resize(Config.img_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])]),
            ),
            batch_size=Config.batch_size,
            shuffle=True,
        )
        self.dataloader = PerpetualLoader(self.dataloader)

    def choochoo(self):
        logging.info('Training started')

        with trange(Status.time + 1, Status.time + Config.training_steps + 1, desc='Steps') as t:
            for Status.time in t:
                assert Config.batch_size['generation'] == Config.batch_size['discrimination']
                assert Config.batches_per_step['generation'] == Config.batches_per_step['discrimination'] == 1

                batch_size = Config.batch_size['generation']

                ones = torch.ones(batch_size, 1)
                z = torch.randn((batch_size, Config.latent_dim))

                gen_imgs = self.models['generator'].forward(z)
                g_loss = self.models['discriminator'].forward(input_imgs=gen_imgs, labels=ones)
                self.reporter.full_report('training', phase, loss, stats)

                self.loose(g_loss, 'generator', 'generation', 1)

                zeros = torch.zeros(batch_size, 1)
                labels = torch.cat([ones, zeros])

                real_imgs = next(self.dataloader)
                input_imgs = torch.cat([gen_imgs, real_imgs])
                d_loss = self.models['discriminator'].forward(input_imgs=input_imgs, labels=labels)

                self.loose(d_loss, 'discriminator', 'discrimination', 1)
