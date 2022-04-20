"""
cgl_fairness
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import torch
import os
from os.path import join
import PIL
import pandas
import random
import zipfile
from functools import partial
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
from torchvision import transforms
import data_handler
from data_handler.utils import get_mean_std


class CelebA(data_handler.SSLDataset):
    """
    There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    right now.
    """
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]
    mean, std = get_mean_std('celeba')
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
    name = 'celeba'

    def __init__(self, target_attr='Attractive', add_attr=None, download=False, **kwargs):
        transform = self.train_transform if kwargs['split'] == 'train' else self.test_transform
        super(CelebA, self).__init__(transform=transform, **kwargs)

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        # SELECT the features
        self.sensitive_attr = 'Male'
        self.add_attr = add_attr
        self.target_attr = target_attr
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[verify_str_arg(self.split.lower(), "split",
                                         ("train", "valid", "test", "all"))]
        if 'group' in self.version:
            split = 0
        fn = partial(join, self.root)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        print('Im add attr : ', target_attr, add_attr)

        mask = slice(None) if split is None else (splits[1] == split)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

        self.target_idx = self.attr_names.index(self.target_attr)
        self.sensi_idx = self.attr_names.index(self.sensitive_attr)
        self.add_idx = self.attr_names.index(self.add_attr) if self.add_attr is not None else -1
        self.feature_idx = [i for i in range(len(self.attr_names)) if i != self.target_idx and i != self.sensi_idx]
        if self.add_attr is not None:
            self.feature_idx.remove(self.add_idx)
        self.num_classes = 2
        self.num_groups = 2 if self.add_attr is None else 4

        if self.add_attr is None:
            self.features = [[int(s), int(l), filename] for s, l, filename in
                             zip(self.attr[:, self.sensi_idx], self.attr[:, self.target_idx], self.filename)]
        else:
            self.features = [[int(s1)*2 + int(s2), int(l), filename] for s1, s2, l, filename in
                             zip(self.attr[:, self.sensi_idx], self.attr[:, self.add_idx], self.attr[:, self.target_idx], self.filename)]
        self.num_data, self.idxs_per_group = self._data_count(self.features, self.num_groups, self.num_classes)

        if self.split == "test" and 'group' not in self.version:
            self.features = self._balance_test_data(self.num_data, self.num_groups, self.num_classes)
            self.num_data, self.idxs_per_group = self._data_count(self.features, self.num_groups, self.num_classes)

        # if semi-supervised learning,
        if self.sv_ratio < 1:
            # we want the different supervision according to the seed
            random.seed(self.seed)
            self.features, self.num_data, self.idxs_per_group = self.ssl_processing(self.features, self.num_data, self.idxs_per_group)
            if 'group' in self.version:
                a, b = self.num_groups, self.num_classes
                self.num_groups, self.num_classes = b, a

    def __getitem__(self, index):
        sensitive, target, img_name = self.features[index]
        image = PIL.Image.open(os.path.join(self.root, "img_align_celeba", img_name))

        if self.transform is not None:
            image = self.transform(image)

        if 'group' in self.version:
            return image, 0, target, sensitive, (index, img_name)
        return image, 0, sensitive, target, (index, img_name)

    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, "img_align_celeba"))

    def _download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, self.root, filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, "img_align_celeba.zip"), "r") as f:
            f.extractall(self.root)
