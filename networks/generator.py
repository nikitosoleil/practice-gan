from torch import nn

from configs import Config


class GeneratorNN(nn.Module):
    filters = [128, 64, 32, 16]

    def __init__(self):
        super(GeneratorNN, self).__init__()
        self.inter_size = Config.img_size // 2 ** (len(GeneratorNN.filters) - 1)
        self.inter_act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.proj = nn.Sequential(nn.Linear(Config.latent_dim, GeneratorNN.filters[0] * self.inter_size ** 2))
        self.upsample = nn.Upsample(scale_factor=2)

        def get_block(in_filters, out_filters):
            return nn.Sequential(self.upsample,
                                 nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(out_filters),
                                 self.inter_act)

        self.blocks = nn.Sequential(*[get_block(inf, outf) for (inf, outf) in
                                      zip(GeneratorNN.filters[:-1], GeneratorNN.filters[1:])])

        self.model = nn.Sequential(
            nn.BatchNorm2d(GeneratorNN.filters[0]),
            self.blocks,
            nn.Conv2d(GeneratorNN.filters[-1], Config.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.proj(z)
        out = out.view(out.shape[0], GeneratorNN.filters[0], self.inter_size, self.inter_size)
        img = self.model(out)

        result = (img,)
        return result
