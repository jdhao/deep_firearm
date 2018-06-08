"""
orignial vgg16 model is fine-tuned to do classification on firearm dataset, the
feature is not normalized.
"""
import math
# from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

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


class VGGNet(nn.Module):
    def __init__(self, features, feature_dim=512):
        super(VGGNet, self).__init__()
        self.features = features
        if feature_dim==512:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 127)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 127)
            )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_max_pool2d(x, output_size=(1, 1))
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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


def vgg16(pretrained=True, feature_dim=512):
    model = VGGNet(make_layers(cfg['D']), feature_dim=feature_dim)

    if pretrained:
        new_state_dict = model.state_dict()

        # print(type(new_state_dict))
        # new_key = [(idx, key) for (idx, key) in enumerate(new_state_dict.keys())]
        # print(new_key)
        old_state_dict = model_zoo.load_url(model_url['vgg16'])

        # old_key = [(idx, key) for (idx, key) in enumerate(old_state_dict.keys())]
        # print(old_key)
        common_key = list(new_state_dict.keys())[:26] # only conv part weigth are useful

        for k in common_key:
            new_state_dict[k] = old_state_dict[k]
        model.load_state_dict(new_state_dict)

    return model