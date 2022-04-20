"""
cgl_fairness
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import numpy as np
import random
from data_handler import SSLDataset


class TabularDataset(SSLDataset):
    """Adult dataset."""
    # 1 idx -> sensi
    # 2 idx -> label
    # 3 idx -> filename or feature (image / tabular)
    def __init__(self, dataset, sen_attr_idx, **kwargs):
        super(TabularDataset, self).__init__(**kwargs)
        self.sen_attr_idx = sen_attr_idx

        dataset_train, dataset_test = dataset.split([0.8], shuffle=True, seed=0)
        # features, labels = self._balance_test_set(dataset)
        self.dataset = dataset_train if (self.split == 'train') or ('group' in self.version) else dataset_test

        features = np.delete(self.dataset.features, self.sen_attr_idx, axis=1)
        mean, std = self._get_mean_n_std(dataset_train.features)
        features = (features - mean) / std

        self.groups = np.expand_dims(self.dataset.features[:, self.sen_attr_idx], axis=1)
        self.labels = np.squeeze(self.dataset.labels)

        # self.features = self.dataset.features
        self.features = np.concatenate((self.groups, self.dataset.labels, features), axis=1)

        # For prepare mean and std from the train dataset
        self.num_data, self.idxs_per_group = self._data_count(self.features, self.num_groups, self.num_classes)

        # if semi-supervised learning,
        if self.sv_ratio < 1:
            # we want the different supervision according to the seed
            random.seed(self.seed)
            self.features, self.num_data, self.idxs_per_group = self.ssl_processing(self.features, self.num_data, self.idxs_per_group, )
            if 'group' in self.version:
                a, b = self.num_groups, self.num_classes
                self.num_groups, self.num_classes = b, a

    def get_dim(self):
        return self.dataset.features.shape[-1]

    def __getitem__(self, idx):
        features = self.features[idx]
        group = features[0]
        label = features[1]
        feature = features[2:]

        if 'group' in self.version:
            return np.float32(feature), 0, label, np.int64(group), (idx, 0)
        else:
            return np.float32(feature), 0, group, np.int64(label), (idx, 0)

    def _get_mean_n_std(self, train_features):
        features = np.delete(train_features, self.sen_attr_idx, axis=1)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] += 1e-7
        return mean, std
