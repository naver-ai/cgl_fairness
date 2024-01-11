"""
cgl_fairness
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import random
import numpy as np
import torch
from data_handler import GenericDataset
import os
import pickle


class SSLDataset(GenericDataset):
    def __init__(self, sv_ratio=1.0, version='', ups_iter=0, **kwargs):
        super(SSLDataset, self).__init__(**kwargs)
        self.sv_ratio = sv_ratio
        self.version = version
        self.add_attr = None
        self.ups_iter = ups_iter

    def ssl_processing(self, features, num_data, idxs_per_group, idxs_dict=None):
        if self.sv_ratio >= 1:
            raise ValueError

        if self.split == 'test' and 'group' not in self.version:
            return features, num_data, idxs_per_group

        print('preprocessing for ssl...')
        num_groups, num_classes = num_data.shape

        folder_name = 'annotated_idxs'
        idx_filename = '{}_{}'.format(self.seed, self.sv_ratio)
        if self.name == 'celeba':
            if self.target_attr != 'Attractive':
                idx_filename += f'_{self.target_attr}'
            if self.add_attr is not None:
                idx_filename += f'_{self.add_attr}'
        idx_filename += '.pkl'
        filepath = os.path.join(self.root, folder_name, idx_filename)

        if idxs_dict is None:
            if not os.path.isfile(filepath):
                idxs_dict = self.pick_idxs(num_data, idxs_per_group, filepath)
            else:
                with open(filepath, 'rb') as f:
                    idxs_dict = pickle.load(f)

        if self.version == 'bs1':
            new_idxs = []
            for g in range(num_groups):
                for l in range(num_classes):
                    new_idxs.extend(idxs_dict['annotated'][(g, l)])
            new_idxs.sort()
            new_features = [features[idx] for idx in new_idxs]
            features = new_features

        elif self.version == 'bs2':
            idx_pool = list(range(num_groups))
            total_per_group = np.zeros((num_classes, num_groups))
            for l in range(num_classes):
                for g in range(num_groups):
                    total_per_group[l, g] = len(idxs_dict['annotated'][(g, l)])
            total_per_group = total_per_group.astype(int)
            for g in range(num_groups):
                for l in range(num_classes):
                    for idx in idxs_dict['non-annotated'][(g, l)]:
                        features[idx][0] = random.choices(idx_pool, k=1, weights=list(total_per_group[l]))[0]

        elif self.version == 'bs1uncertain':
            # bs2new is to use unlabeled data only for accuracy
            folder = 'group_clf'
            model = 'resnet18_dropout'
            epochs = '70'
            filename_pre = f'{model}_seed{self.seed}_epochs{epochs}_bs128_lr0.001'
            filename_post = f'_sv{self.sv_ratio}_groupclf.pt'
            filename = filename_pre + filename_post
            path = os.path.join(self.root, folder, filename)
            preds = torch.load(path)['pred']

            new_idxs = []
            for g in range(num_groups):
                for l in range(num_classes):
                    # annotated
                    new_idxs.extend(idxs_dict['annotated'][(g, l)])
                    # non-annotated
                    for idx in idxs_dict['non-annotated'][(g, l)]:
                        if preds[idx] != -1:
                            new_idxs.append(idx)
                            features[idx][0] = preds[idx].item()

            print(len(new_idxs), len(features))
            new_idxs.sort()
            new_features = [features[idx] for idx in new_idxs]
            features = new_features

        elif self.version == 'bs3':
            # bs 3 is to make psuedo labels using a model being trained with labeld group label from scratch
            folder = 'group_clf' if self.version == 'bs3' else 'group_clf_pretrain'
            model = 'resnet18' if self.name in ['utkface', 'celeba'] else 'mlp'
            epochs = '70' if self.name in ['utkface', 'celeba'] else '50'
            filename_pre = f'{model}_seed{self.seed}_epochs{epochs}_bs128_lr0.001'
            filename_post = f'_sv{self.sv_ratio}_groupclf.pt'
            if self.name == 'celeba':
                if self.target_attr != 'Attractive':
                    filename_pre += f'_{self.target_attr}'
                if self.add_attr is not None:
                    filename_pre += f'_{self.add_attr}'
            filename = filename_pre + filename_post
            path = os.path.join(self.root, folder, filename)
            preds = torch.load(path)['pred']

            for g in range(num_groups):
                for l in range(num_classes):
                    for idx in idxs_dict['non-annotated'][(g, l)]:
                        features[idx][0] = preds[idx].item()

        elif self.version == 'groupclf':
            if self.ups_iter > 0:
                folder = 'group_clf'
                filename_pre = f'resnet18_dropout_seed{self.seed}_epochs70_bs128_lr0.001'
                if self.ups_iter > 1:
                    filename_post = f'_sv{self.sv_ratio}_iter{_iter}_groupclf.pt'
                else:
                    filename_post = f'_sv{self.sv_ratio}_groupclf.pt'
                filename = filename_pre + filename_post
                path = os.path.join(self.root, folder, filename)
                preds = torch.load(path)['pred']

            new_idxs = []
            for g in range(num_groups):
                for l in range(num_classes):
                    if self.split == 'train':
                        idxs = idxs_dict['annotated'][(g, l)]
                    elif self.split == 'test':
                        idxs = idxs_dict['non-annotated'][(g, l)]
                    new_idxs.extend(idxs)
            new_idxs.sort()
            new_features = [features[idx] for idx in new_idxs]
            features = new_features

        elif self.version == 'groupclf_val':
            new_idxs = []
            val_idxs = []
            for g in range(num_groups):
                for l in range(num_classes):
                    if self.split == 'train':
                        train_num = int(len(idxs_dict['annotated'][(g, l)]) * 0.8)
                        idxs = idxs_dict['annotated'][(g, l)][:train_num]
                        val_idxs.extend(idxs_dict['annotated'][(g, l)][train_num:])
                    elif self.split == 'test':
                        idxs = idxs_dict['non-annotated'][(g, l)]
                    new_idxs.extend(idxs)
            new_idxs.sort()
            new_features = [features[idx] for idx in new_idxs]
            features = new_features
            self.val_idxs = val_idxs

        elif self.version == 'oracle':
            # this version is to make a oracle model about predicting noisy lables.
            folder = 'group_clf'
            filename = 'resnet18_seed{}_epochs70_bs128_lr0.001_sv{}_groupclf_val.pt'.format(self.seed, self.sv_ratio)
            path = os.path.join(self.root, folder, filename)
            preds = torch.load(path)['pred']
            idx_pool = list(range(num_groups))
            for g in range(num_groups):
                for l in range(num_classes):
                    for idx in idxs_dict['non-annotated'][(g, l)]:
                        if features[idx][0] != preds[idx].item():
                            features[idx][0] = random.choices(idx_pool, k=1)[0]

        elif self.version == 'cgl':
            folder = 'group_clf'
            model = 'resnet18' if self.name in ['utkface', 'celeba', 'utkface_fairface'] else 'mlp'
            epochs = '70' if self.name in ['utkface', 'celeba', 'utkface_fairface'] else '50'
            bs = '128' if self.name != 'adult' else '1024'
            filename_pre = f'{model}_seed{self.seed}_epochs{epochs}_bs{bs}_lr0.001'
            filename_post = f'_sv{self.sv_ratio}_groupclf_val.pt'
            if self.name == 'celeba':
                if self.target_attr != 'Attractive':
                    filename_pre += f'_{self.target_attr}'
                if self.add_attr is not None:
                    filename_pre += f'_{self.add_attr}'
            filename = filename_pre + filename_post

            if self.name == 'utkface_fairface':
                path = os.path.join('./data/utkface_fairface', folder, filename)
            else:
                path = os.path.join(self.root, folder, filename)

            preds = torch.load(path)['pred']
            probs = torch.load(path)['probs']
            thres = torch.load(path)['opt_thres']
            print('thres : ', thres)
            idx_pool = list(range(num_groups))

            total_per_group = np.zeros((num_classes, num_groups))
            for l in range(num_classes):
                for g in range(num_groups):
                    total_per_group[l, g] = len(idxs_dict['annotated'][(g, l)])
            total_per_group = total_per_group.astype(int)
            for g in range(num_groups):
                for l in range(num_classes):
                    for idx in idxs_dict['non-annotated'][(g, l)]:
                        if self.version == 'cgl':
                            if probs[idx].max() >= thres:
                                features[idx][0] = preds[idx].item()
                            else:
                                features[idx][0] = random.choices(idx_pool, k=1, weights=list(total_per_group[l]))[0]

        else:
            raise ValueError
        print('count the number of data newly!')
        num_data, idxs_per_group = self._data_count(features, num_groups, num_classes)

        return features, num_data, idxs_per_group

    def pick_idxs(self, num_data, idxs_per_group, filepath):
        print('<pick idxs : {}>'.format(filepath))
        if not os.path.isdir(os.path.join(self.root, 'annotated_idxs')):
            os.mkdir(os.path.join(self.root, 'annotated_idxs'))
        num_groups, num_classes = num_data.shape
        idxs_dict = {}
        idxs_dict['annotated'] = {}
        idxs_dict['non-annotated'] = {}
        for g in range(num_groups):
            for l in range(num_classes):
                num_nonannotated = int(num_data[g, l] * (1-self.sv_ratio))
                print(g, l, num_nonannotated)
                idxs_nonannotated = random.sample(idxs_per_group[(g, l)], num_nonannotated)
                idxs_annotated = [idx for idx in idxs_per_group[(g, l)] if idx not in idxs_nonannotated]
                idxs_dict['non-annotated'][(g, l)] = idxs_nonannotated
                idxs_dict['annotated'][(g, l)] = idxs_annotated

        with open(filepath, 'wb') as f:
            pickle.dump(idxs_dict, f, pickle.HIGHEST_PROTOCOL)

        return idxs_dict
