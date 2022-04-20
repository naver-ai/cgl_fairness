"""
cgl_fairness
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import pandas as pd
from data_handler.AIF360.adult_dataset import AdultDataset
from data_handler.tabular_dataset import TabularDataset


class AdultDataset_torch(TabularDataset):
    """Adult dataset."""
    name = 'adult'
    def __init__(self, root, target_attr='sex', **kwargs):

        dataset = AdultDataset(root_dir=root)
        if target_attr == 'sex':
            sen_attr_idx = 3
        elif target_attr == 'race':
            sen_attr_idx = 2
        else:
            raise Exception('Not allowed group')

        self.num_groups = 2
        self.num_classes = 2

        super(AdultDataset_torch, self).__init__(root=root, dataset=dataset, sen_attr_idx=sen_attr_idx,
                                                 **kwargs)

