import torch
from torch import nn

from configs import Config


class DiscriminatorNN(nn.Module):
    filters = [Config.channels, 16, 32, 64, 128]

    def __init__(self):
        super().__init__()
        self.inter_size = Config.img_size // 2 ** (len(DiscriminatorNN.filters) - 1)
        self.inter_act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.dropout = nn.Dropout2d(p=0.25)

        def get_block(in_filters, out_filters):
            return nn.Sequential(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
                                 self.inter_act,
                                 self.dropout,
                                 nn.BatchNorm2d(out_filters))

        self.blocks = nn.Sequential(*[get_block(inf, outf) for (inf, outf) in
                                      zip(DiscriminatorNN.filters[:-1], DiscriminatorNN.filters[1:])])

        self.proj = nn.Linear(DiscriminatorNN.filters[-1] * self.inter_size ** 2, 1)

    def forward(self, input_imgs, labels=None):
        out = self.blocks(input_imgs)
        out = out.view(out.shape[0], -1)
        logits = self.proj(out)

        result = (logits,)
        return result


class GANDiscriminatorNN(DiscriminatorNN):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_imgs, labels=None):
        result = super().forward(input_imgs, labels)
        logits = result[0]
        if labels is not None:
            loss = self.loss(logits, labels)
            result = (loss,) + result
        return result
