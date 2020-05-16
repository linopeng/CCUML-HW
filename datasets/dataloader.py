from .transforms import build_transform
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import os
from pathlib import Path
from hw2.config import cfg
from PIL import Image

root_path = cfg.PATH.TRAIN_SET

label_dict = {
    'A': 0,
    'B': 1,
    'C': 2
}


class MangoDataset(Dataset):
    def __init__(self, _dir, data_transform=None):
        self.root_dir = Path(_dir)
        self.x = []
        self.y = []
        self.data_transform = data_transform
        if _dir.name == 'C1-P1_Train':
            labels = np.genfromtxt(Path(root_path).joinpath('train.csv'), dtype=np.str, delimiter=',', skip_header=1)
        else:
            labels = np.genfromtxt(Path(root_path).joinpath('dev.csv'), dtype=np.str, delimiter=',', skip_header=1)

        for label in labels:
            self.x.append(label[0])
            self.y.append(label_dict[label[1]])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image_path = Path(self.root_dir).joinpath(self.x[index])
        image = Image.open(image_path).convert('RGB')

        if self.data_transform:
            image = self.data_transform(image)
        return image, self.y[index]

    """transforms = build_transform(cfg)

    trainset = datasets.ImageFolder(train_path, transform=transforms)
    
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)"""
    # train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    # valid_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, sampler=valid_sampler)
    # return train_loader, valid_loader


def make_test_loader(cfg):
    num_workers = cfg.DATA.NUM_WORKERS
    batch_size = cfg.DATA.TEST_BATCH_SIZE
    test_path = cfg.PATH.TEST_SET

    transforms = build_transform(cfg)

    testset = datasets.ImageFolder(test_path, transform=transforms)

    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)

    return test_loader
