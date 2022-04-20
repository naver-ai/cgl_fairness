"""
Original code:
    https://github.com/sangwon79/Fair-Feature-Distillation-for-Visual-Recognition
"""
from copy import copy
import numpy as np
from torch.utils.data.sampler import RandomSampler
import random


class Customsampler(RandomSampler):

    def __init__(self, data_source, replacement=False, num_samples=None, batch_size=None, generator=None):
        super(Customsampler, self).__init__(data_source=data_source, replacement=replacement,
                                            num_samples=num_samples, generator=generator)
        self.l = data_source.num_classes
        self.g = data_source.num_groups
        self.nbatch_size = batch_size // (self.l*self.g)
        self.num_data = data_source.num_data
        self.idxs_per_group = data_source.idxs_per_group

        # which one is a group that has the largest number of data poitns
        self.max_pos = np.unravel_index(np.argmax(self.num_data), self.num_data.shape)
        self.numdata_per_group = (self.num_data[self.max_pos] // (self.nbatch_size+1) + 1) * (self.nbatch_size+1)

    def __iter__(self):
        index_list = []

        for g in range(self.g):
            for l in range(self.l):
                total = 0
                group_index_list = []
                while total < self.numdata_per_group:
                    tmp = copy(self.idxs_per_group[(g, l)])
                    random.shuffle(tmp)
                    remained_data = self.numdata_per_group - total
                    if remained_data > len(tmp):
                        group_index_list.extend(tmp)
                    else:
                        group_index_list.extend(tmp[:remained_data])
                        break
                    total += len(tmp)
                index_list.append(group_index_list)

        final_list = np.array(index_list)
        final_list = final_list.flatten('F')
        final_list = list(final_list)

        return iter(final_list)
