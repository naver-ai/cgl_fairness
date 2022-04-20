"""
cgl_fairness
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import torch
from torch.utils.data import DataLoader
import numpy as np


def get_mean_std(dataset, skew_ratio=0.8):
    mean, std = None, None

    if dataset == 'utkface':
        mean = [0.5960, 0.4573, 0.3921]
        std = [0.2586, 0.2314, 0.2275]

    elif dataset == 'cifar10s':
        # for skew 0.8
        mean = [0.4871, 0.4811, 0.4632]
        std = [0.2431, 0.2414, 0.2506]

    elif dataset == 'celeba':
        # default target is 'Attractive'
        mean = [0.5063, 0.4258, 0.3832]
        std = [0.3107, 0.2904, 0.2897]

    elif dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return mean, std


def predict_group(model, loader, args):
    model.cuda('cuda:{}'.format(args.device))
    if args.slversion == 3:
        filename = 'trained_models/group_clf/utkface/scratch/resnet18_seed{}_epochs70_bs128_lr0.001_sv{}_version0.0.pt'
    elif args.slversion == 5:
        filename = 'trained_models/group_clf_pretrain/utkface/scratch/resnet18_seed{}_epochs70_bs128_lr0.001_sv{}_version0.0.pt'
    path = filename.format(str(args.seed), str(args.sv))
    model.load_state_dict(torch.load(path, map_location=torch.device('cuda:{}'.format(args.device))))

    features = loader.dataset.features

    dataloader = DataLoader(loader.dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, _, groups, labels, (idxs, _) = data
            if (groups == -1).sum() == 0:
                continue

            if args.cuda:
                inputs = inputs.cuda()
                groups = groups.cuda()
                idxs = idxs.cuda()
            inputs = inputs[groups == -1]
            idxs = idxs[groups == -1]

            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            for j, idx in enumerate(idxs.cpu().numpy()):
                features[idx][1] = preds.cpu()[j]
            if i % args.term == 0:
                print('[{}] in group prediction'.format(i))

    if args.labelwise:
        loader.dataset.num_data, loader.dataset.idxs_per_group = loader.dataset._data_count()
        from data_handler.custom_loader import Customsampler

        def _init_fn(worker_id):
            np.random.seed(int(args.seed))
        sampler = Customsampler(loader.dataset, replacement=False, batch_size=args.batch_size)
        train_dataloader = DataLoader(loader.dataset, batch_size=args.batch_size, sampler=sampler,
                                      num_workers=args.num_workers, worker_init_fn=_init_fn, pin_memory=True, drop_last=True)

    del dataloader
    del model
    del loader
    if args.labelwise:
        return train_dataloader
