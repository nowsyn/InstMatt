import torch
import torch.nn as nn

from utils import CONFIG, concat_all_gather, group_reduce_sum, reduce_dict, reduce_list
from networks import encoders, decoders, refiners, ops

class Generator(nn.Module):
    def __init__(self, cfg, encoder, decoder, cross_head=False, refiner=None):

        super(Generator, self).__init__()

        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        self.encoder = encoders.__dict__[encoder]()

        self.aspp = ops.ASPP(in_channel=512, out_channel=512)

        if decoder not in decoders.__all__:
            raise NotImplementedError("Unknown Decoder {}".format(decoder))
        self.decoder = decoders.__dict__[decoder](act_func=cfg.model.act_function)

        if refiner is not None:
            if cfg.model.freeze:
                for p in self.parameters():
                    p.requires_grad = False
                print("Freeze weights of the parameters in the backbone")
            self.refiner = refiners.__dict__[refiner]()
        else:
            self.refiner = None

        if cross_head:
            for p in self.parameters():
                p.requires_grad=False

            self.cross_head = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 1, kernel_size=3, padding=1)
            )
        else:
            self.cross_head = None

    def forward(self, image, guidance, is_training=True):
        inp = torch.cat((image, guidance), dim=1)
        embedding, mid_fea = self.encoder(inp)
        embedding = self.aspp(embedding)
        pred = self.decoder(embedding, mid_fea, is_training=is_training)
        return pred

    def forward_refiner(self, x, pred, feat, is_training=True, nostop=True):
        return self.refiner(x, pred, feat, is_training, nostop=nostop)


def get_generator(cfg, encoder, decoder, refiner=None):
    generator = Generator(cfg, encoder=encoder, decoder=decoder, refiner=refiner)
    return generator
