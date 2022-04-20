"""
cgl_fairness
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import importlib
import torch.utils.data as data
import numpy as np
from collections import defaultdict

dataset_dict = {
    'utkface': ['data_handler.utkface', 'UTKFaceDataset'],
    'utkface_fairface': ['data_handler.utkface_fairface', 'UTKFaceFairface_Dataset'],
    'celeba': ['data_handler.celeba', 'CelebA'],
    'adult': ['data_handler.adult', 'AdultDataset_torch'],
    'compas': ['data_handler.compas', 'CompasDataset_torch'],
    'cifar100s': ['data_handler.cifar100s', 'CIFAR_100S'],
}


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, split='Train', seed=0, sv_ratio=1, version=1, target_attr='Attractive', add_attr=None, ups_iter=0):
        root = f'./data/{name}' if name != 'utkface_fairface' else './data/utkface'
        kwargs = {
            'root': root,
            'split': split,
            'seed': seed,
            'sv_ratio': sv_ratio,
            'version': version,
            'ups_iter': ups_iter
        }

        if name not in dataset_dict.keys():
            raise Exception('Not allowed method')

        if name == 'celeba':
            kwargs['add_attr'] = add_attr
            kwargs['target_attr'] = target_attr
        elif name == 'adult':
            kwargs['target_attr'] = target_attr
        elif name == 'compas':
            kwargs['target_attr'] = target_attr

        module = importlib.import_module(dataset_dict[name][0])
        class_ = getattr(module, dataset_dict[name][1])
        return class_(**kwargs)


class GenericDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None, seed=0):
        self.root = root
        self.split = split
        self.transform = transform
        self.seed = seed
        self.num_data = None

    def __len__(self):
        return np.sum(self.num_data)

    def _data_count(self, features, num_groups, num_classes):
        idxs_per_group = defaultdict(lambda: [])
        data_count = np.zeros((num_groups, num_classes), dtype=int)

        for idx, i in enumerate(features):
            s, l = int(i[0]), int(i[1])
            data_count[s, l] += 1
            idxs_per_group[(s, l)].append(idx)

        print(f'mode : {self.split}')
        for i in range(num_groups):
            print('# of %d group data : ' % i, data_count[i, :])
        return data_count, idxs_per_group

    def _make_data(self, features, num_groups, num_classes):
        # if the original dataset not is divided into train / test set, this function is used
        min_cnt = 100
        data_count = np.zeros((num_groups, num_classes), dtype=int)
        tmp = []
        for i in reversed(self.features):
            s, l = int(i[0]), int(i[1])
            data_count[s, l] += 1
            if data_count[s, l] <= min_cnt:
                features.remove(i)
                tmp.append(i)

        train_data = features
        test_data = tmp
        return train_data, test_data

    def _make_weights(self):
        group_weights = len(self) / self.num_data
        weights = [group_weights[s, l] for s, l, _ in self.features]
        return weights

    def _balance_test_data(self, num_data, num_groups, num_classes):
        print('balance test data...')
        # if the original dataset is divided into train / test set, this function is used
        num_data_min = np.min(num_data)
        print('min : ', num_data_min)
        data_count = np.zeros((num_groups, num_classes), dtype=int)
        new_features = []
        for idx, i in enumerate(self.features):
            s, l = int(i[0]), int(i[1])
            if data_count[s, l] < num_data_min:
                new_features.append(i)
                data_count[s, l] += 1

        return new_features
