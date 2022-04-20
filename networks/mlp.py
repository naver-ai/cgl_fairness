"""
cgl_fairness
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class MLP(nn.Module):
    def __init__(self, feature_size, hidden_dim, num_classes=None, num_layer=3, adv=False, adv_lambda=1.):
        super(MLP, self).__init__()
        try: #list
            in_features = self.compute_input_size(feature_size)
        except : #int
            in_features = feature_size

        num_outputs = num_classes
        self.adv = adv
        if self.adv:
            self.adv_lambda = adv_lambda
        self._make_layer(in_features, hidden_dim, num_classes, num_layer)

    def forward(self, feature, get_inter=False):
        feature = torch.flatten(feature, 1)
        if self.adv:
            feature = ReverseLayerF.apply(feature, self.adv_lambda)

        h = self.features(feature)
        out = self.head(h)
        out = out.squeeze()

        if get_inter:
            return h, out
        else:
            return out

    def compute_input_size(self, feature_size):
        in_features = 1
        for size in feature_size:
            in_features = in_features * size

        return in_features

    def _make_layer(self, in_dim, h_dim, num_classes, num_layer):

        if num_layer == 1:
            self.features = nn.Identity()
            h_dim = in_dim
        else:
            features = []
            for i in range(num_layer-1):
                features.append(nn.Linear(in_dim, h_dim) if i == 0 else nn.Linear(h_dim, h_dim))
                features.append(nn.ReLU())
            self.features = nn.Sequential(*features)

        self.head = nn.Linear(h_dim, num_classes)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
