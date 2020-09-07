import logging
from typing import Dict

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
from utils.reporter import Reporter
from utils.scheduler import Scheduler
from .locomotive import Locomotive


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class GANTrain(Locomotive):
    def __init__(self, dataset: Dataset, models: Dict[str, BaseModel], optimizers: Dict[str, Optimizer],
                 reporter: Reporter, checkpointer: Checkpointer):
        super().__init__(dataset, models, optimizers, reporter, checkpointer)

        for model in self.models:
            self.models[model].network.apply(weights_init_normal)

        assert Config.batch_size['generation'] == Config.batch_size['discrimination']  # single dataloader

        self.dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                Config.data_path, train=True, download=True,
                transform=transforms.Compose([transforms.Resize(Config.img_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])]),
            ),
            batch_size=Config.batch_size['generation'],
            shuffle=True,
            drop_last=True
        )
        self.dataloader = PerpetualLoader(self.dataloader)

    def choochoo(self):
        logging.info('Training started')

        with trange(Status.time + 1, Status.time + Config.training_steps + 1, desc='Steps') as t:
            for Status.time in t:
                assert Config.batch_size['generation'] == Config.batch_size['discrimination']  # gen_imgs reusage
                batch_size = Config.batch_size['generation']

                assert Config.batches_per_step['generation'] == Config.batches_per_step['discrimination'] == 1  # no loops
                assert Config.accumulation_steps['generation'] == Config.accumulation_steps['discrimination'] == 1  # .zero_grad()

                if Config.is_running['generation']:
                    # t.set_postfix_str('generation')

                    self.optimizers['generator'].zero_grad()

                    ones = torch.ones(batch_size, 1)
                    z = torch.randn((batch_size, Config.latent_dim))

                    gen_imgs = self.models['generator'].train(z)[0]

                    g_loss, g_stats = self.models['discriminator'].train(input_imgs=gen_imgs, labels=ones)

                    self.loose(g_loss, 'generator', 'generation', 1)
                    self.reporter.full_report('training', 'generation', g_loss, g_stats)

                if Config.is_running['discrimination']:
                    # t.set_postfix_str('discrimination')

                    self.optimizers['discriminator'].zero_grad()

                    real_imgs = next(self.dataloader)[0].to(gen_imgs.device)
                    zeros = torch.zeros(batch_size, 1)
                    gen_imgs = gen_imgs.detach()

                    real_loss = self.models['discriminator'].train(real_imgs, ones)[0]
                    fake_loss = self.models['discriminator'].train(gen_imgs, zeros)[0]
                    d_loss = (real_loss + fake_loss) / 2

                    self.loose(d_loss, 'discriminator', 'discrimination', 1)
                    self.reporter.full_report('training', 'discrimination', d_loss, {})

                if Scheduler.is_logging():
                    self.reporter.push('training')

                if Scheduler.is_validating('generator'):
                    save_image(gen_imgs.data[:25], "images/%d.png" % Status.time, nrow=5, normalize=True)

                if Scheduler.is_checkpointing() and not Config.test_mode:
                    self.checkpointer.save(self.models, self.optimizers, Status.time)
