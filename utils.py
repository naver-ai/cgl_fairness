"""
Original code:
    https://github.com/sangwon79/Fair-Feature-Distillation-for-Visual-Recognition
"""
import torch
import numpy as np
import random
import os
import torch.nn.functional as F


def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files


def set_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_accuracy(outputs, labels, binary=False, reduction='mean'):
    # if multi-label classification
    if len(labels.size()) > 1:
        outputs = (outputs > 0.0).float()
        correct = ((outputs == labels)).float().sum()
        total = torch.tensor(labels.shape[0] * labels.shape[1], dtype=torch.float)
        avg = correct / total
        return avg.item()

    if binary:
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
    else:
        predictions = torch.argmax(outputs, 1)

    c = (predictions == labels).float().squeeze()
    if reduction == 'none':
        return c
    else:
        accuracy = torch.mean(c)
        return accuracy.item()


def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
        print("Failed to create directory!!")


class FitnetRegressor(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super(FitnetRegressor, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.regressor = torch.nn.Conv2d(in_feature, out_feature, 1, bias=False)
        torch.nn.init.kaiming_normal_(self.regressor.weight, mode='fan_out', nonlinearity='relu')
        self.regressor.weight.data.uniform_(-0.005, 0.005)

    def forward(self, feature):
        if feature.dim() == 2:
            feature = feature.unsqueeze(2).unsqueeze(3)

        return F.relu(self.regressor(feature))


def make_log_name(args):
    log_name = args.model

    if args.mode == 'eva':
        log_name = args.modelpath.split('/')[-1]
        # remove .pt from name
        log_name = log_name[:-3]

    else:
        if args.pretrained:
            log_name += '_pretrained'
        log_name += '_seed{}_epochs{}_bs{}_lr{}'.format(args.seed, args.epochs, args.batch_size, args.lr)
        if args.method == 'adv':
            log_name += '_lamb{}_eta{}'.format(args.lamb, args.eta)

        elif args.method == 'scratch_mmd' or args.method == 'kd_mfd':
            log_name += '_{}'.format(args.kernel)
            log_name += '_sigma{}'.format(args.sigma) if args.kernel == 'rbf' else ''
            log_name += '_{}'.format(args.lambhf)

        elif args.method == 'reweighting':
            log_name += '_eta{}_iter{}'.format(args.eta, args.iteration)

        elif 'groupdro' in args.method:
            log_name += '_gamma{}'.format(args.gamma)

        if args.labelwise:
            log_name += '_labelwise'

        if args.teacher_path is not None or args.method == 'fairhsic':
            log_name += '_lamb{}'.format(args.lamb)
            log_name += '_from_{}'.format(args.teacher_type)

        if args.dataset == 'celeba':
            if args.target != 'Attractive':
                log_name += '_{}'.format(args.target)
            if args.add_attr is not None:
                log_name += '_{}'.format(args.add_attr)
        if args.sv < 1:
            log_name += '_sv{}'.format(args.sv)
            log_name += '_{}'.format(args.version)

    return log_name
