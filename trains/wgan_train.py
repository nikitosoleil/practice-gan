import logging

import torch
from torchvision.utils import save_image
from tqdm import trange

from configs import Config, Status
from utils.scheduler import Scheduler
from . import GANTrain


class WGANTrain(GANTrain):
    def choochoo(self):
        logging.info('Training started')

        with trange(Status.time + 1, Status.time + Config.training_steps + 1, desc='Steps') as t:
            for Status.time in t:

                phase = 'generation'

                if Config.is_running[phase]:
                    # t.set_postfix_str(phase)
                    for i in range(Config.batches_per_step[phase]):
                        self.optimizers['generator'].zero_grad()

                        z = torch.randn((Config.batch_size[phase], Config.latent_dim))
                        gen_imgs = self.models['generator'].train(z)[0]
                        fake_loss, _ = self.models['discriminator'].train(input_imgs=gen_imgs)
                        g_loss = - fake_loss

                        self.loose(g_loss, 'generator', phase, 1)
                        self.reporter.full_report('training', phase, g_loss, {})

                phase = 'discrimination'

                if Config.is_running[phase]:
                    # t.set_postfix_str(phase)
                    for i in range(Config.batches_per_step[phase]):

                        self.optimizers['discriminator'].zero_grad()

                        z = torch.randn((Config.batch_size[phase], Config.latent_dim))
                        gen_imgs = self.models['generator'].train(z)[0]

                        real_imgs = next(self.dataloader)[0].to(gen_imgs.device)
                        gen_imgs = gen_imgs.detach()

                        real_loss = self.models['discriminator'].train(real_imgs)[0]
                        fake_loss = self.models['discriminator'].train(gen_imgs)[0]
                        d_loss = - real_loss + fake_loss

                        self.loose(d_loss, 'discriminator', phase, 1)

                        for name, p in self.models['discriminator'].network.named_parameters():
                            p.data.clamp_(-Config.clip_value, Config.clip_value)

                        self.reporter.full_report('training', phase, d_loss, {})

                if Scheduler.is_logging():
                    self.reporter.push('training')

                if Scheduler.is_validating('generator'):
                    self.reporter.direct_report('sampling', gen_imgs[:Config.val_samples])
                    # TODO: remove this
                    save_image(gen_imgs[:Config.val_samples], "images_wgan/%d.png" % Status.time, nrow=5, normalize=True)

                if Scheduler.is_checkpointing() and not Config.test_mode:
                    self.checkpointer.save(self.models, self.optimizers, Status.time)
