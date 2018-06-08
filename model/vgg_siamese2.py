"""
this script is used to provide retrieval model.

use model fine-tuned on firearm dataset using classification task instead of
original vgg pretrained on ImageNet
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils import model_zoo
from model.vgg_cls_net2 import vgg16


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
model_url = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGBaseNet(nn.Module):
    def __init__(self, features, feature_dim):
        super(VGGBaseNet, self).__init__()
        self.features = features
        self.feature_dim = feature_dim

        if self.feature_dim != 512:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, feature_dim),
                nn.ReLU(inplace=True)
            )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_max_pool2d(x, output_size=(1, 1))
        x = x.view(x.size(0), -1)

        if self.feature_dim == 512:
            return F.normalize(x, p=2, dim=1)
        else:
            x = self.classifier(x)
            return F.normalize(x, p=2, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg16_basenet(pretrained=True,
                  checkpoint_dir=None,
                  feature_dim=512):
    """
    :param pretrained: whether to use model pretrained firearm classification
    :param checkpoint_dir: the classification model checkpoint directory
    :param feature_dim: feature diemension of classification model
    :return: retreival base model which will produce the feature embedding for an image
    """
    # base model of retrieval
    embeding_net = VGGBaseNet(make_layers(cfg['D']), feature_dim=feature_dim)

    # we use the classification model to initialize the retrieval base model
    model_cls = vgg16(pretrained=True, feature_dim=feature_dim)

    if pretrained:
        checkpoint = torch.load(checkpoint_dir)
        model_cls.load_state_dict(checkpoint['state_dict'])
        old_state_dict = model_cls.state_dict()

        new_state_dict = embeding_net.state_dict()

        for k in old_state_dict:
            if k in new_state_dict:
                new_state_dict[k] = old_state_dict[k]
        embeding_net.load_state_dict(new_state_dict)

    return embeding_net


class SiameseNetBaseline(nn.Module):
    def __init__(self, embeddingnet):
        super(SiameseNetBaseline, self).__init__()
        self.embeddingnet = embeddingnet

    def forward_once(self, x):
        return self.embeddingnet(x)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        return out1, out2

