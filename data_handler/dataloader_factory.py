"""
Original code:
    https://github.com/sangwon79/Fair-Feature-Distillation-for-Visual-Recognition
"""
from data_handler.dataset_factory import DatasetFactory

import numpy as np
from torch.utils.data import DataLoader


class DataloaderFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataloader(name, batch_size=256, seed=0, num_workers=4,
                       target_attr='Attractive', add_attr=None, labelwise=False, sv_ratio=1, version='', args=None):
        if name == 'adult':
            target_attr = 'sex'
        elif name == 'compas':
            target_attr = 'race'

        test_dataset = DatasetFactory.get_dataset(name, split='test', sv_ratio=sv_ratio, version=version,
                                                  target_attr=target_attr, seed=seed, add_attr=add_attr)
        train_dataset = DatasetFactory.get_dataset(name, split='train', sv_ratio=sv_ratio, version=version,
                                                   target_attr=target_attr, seed=seed, add_attr=add_attr)

        def _init_fn(worker_id):
            np.random.seed(int(seed))

        num_classes = test_dataset.num_classes
        num_groups = test_dataset.num_groups

        shuffle = True
        sampler = None
        if labelwise:
            if args.method == 'fairhsic':
                from data_handler.custom_loader_hsic import Customsampler
                sampler = Customsampler(train_dataset, replacement=False, batch_size=batch_size)
            else:
                from data_handler.custom_loader import Customsampler
                sampler = Customsampler(train_dataset, replacement=False, batch_size=batch_size)

            shuffle = False

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                      num_workers=num_workers, worker_init_fn=_init_fn, pin_memory=True, drop_last=True)

        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                                     num_workers=num_workers, worker_init_fn=_init_fn, pin_memory=True)

        print('# of test data : {}'.format(len(test_dataset)))
        print('# of train data : {}'.format(len(train_dataset)))
        print('Dataset loaded.')
        print('# of classes, # of groups : {}, {}'.format(num_classes, num_groups))

        return num_classes, num_groups, train_dataloader, test_dataloader
