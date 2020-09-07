import torch
from torch import nn

from configs import Config


class DiscriminatorNN(nn.Module):
    def __init__(self):
        super(DiscriminatorNN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(Config.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = Config.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

        self.loss = torch.nn.BCELoss()

    def forward(self, input_imgs, labels=None):
        out = self.model(input_imgs)
        out = out.view(out.shape[0], -1)
        probs = self.adv_layer(out)
        result = (probs,)
        if labels is not None:
            loss = self.loss(probs, labels)
            result = (loss,) + result
        return result
