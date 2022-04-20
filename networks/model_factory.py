"""
cgl_fairness
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import torch.nn as nn

from networks.resnet import resnet10, resnet12,resnet18, resnet34, resnet50, resnet101
from networks.mlp import MLP
from networks.resnet_dropout import resnet18_dropout

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(target_model, num_classes=2, img_size=224, pretrained=False, num_groups=2):

        if target_model == 'mlp':
            return MLP(feature_size=img_size, hidden_dim=64, num_classes=num_classes)

        elif 'resnet' in target_model:
            model_class = eval(target_model)
            if pretrained:
                model = model_class(pretrained=True, img_size=img_size)
                model.fc = nn.Linear(in_features=model.fc.weight.shape[1], out_features=num_classes, bias=True)
            else:
                model = model_class(pretrained=False, num_classes=num_classes, num_groups=num_groups, img_size=img_size)
            return model

        else:
            raise NotImplementedError

