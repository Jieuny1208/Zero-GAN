import torch
import torch.nn as nn
import torch.nn.parallel
from models.networks.encoder import GANEncoder

class Discriminator(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        model = GANEncoder(opt.isize, 1, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features
