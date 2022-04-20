"""
cgl_fairness
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import os
from os.path import join
from data_handler import GenericDataset
import pandas
from torchvision import transforms
from functools import partial
from data_handler.utils import get_mean_std


class Fairface(GenericDataset):
    mean, std = get_mean_std('uktface')
    train_transform = transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)]
        )
    test_transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)]
        )
    name = 'fairface'

    def __init__(self, target_attr='age', root='fairface', split='train', seed=0):
        transform = self.train_transform if split == 'train' else self.test_transform
        GenericDataset.__init__(self, root=root, split=split, seed=seed, transform=transform)

        self.sensitive_attr = 'race'
        self.target_attr = target_attr

        fn = partial(join, self.root)
        label_file = "fairface_label_{}.csv".format(split)
        label_mat = pandas.read_csv(fn(label_file))
        self.feature_mat = self._preprocessing(label_mat)

    def _preprocessing(self, label_mat):
        race_dict = {
            'White': 0,
            'Middle Eastern': 0,
            'Black': 1,
            'East Asian': 2,
            'Southeast Asian': 2,
            'Indian': 3
        }

        age_dict = {
            '0-2': 0,
            '3-9': 0,
            '10-19': 0,
            '20-29': 1,
            '30-39': 1,
            '40-49': 2,
            '50-59': 2,
            '60-69': 2,
            'more than 70': 2,
        }

        race_idx = 3
        age_idx = 1

        feature_mat = []
        for row in label_mat.values:
            _age = row[age_idx]
            _race = row[race_idx]
            if _race not in race_dict.keys():
                continue
            race = race_dict[_race]
            age = age_dict[_age]
            feature_mat.append([race, age, os.path.join(self.root, row[0])])

        return feature_mat
