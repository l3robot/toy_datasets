"""
This module defines the SinMixDataset pytorch class
---------------------------------------------------

author: Louis-Émile Robitaille (l3robot)
date-of-creation: 2018-10-11
"""

import os
import json

import torch
from torch.utils.data import Dataset


class SinMixDataset(Dataset):
    """
    Defines the SinMinDataset
    -------------------------

    Args:
        root (str): location of the dataset
        train (bool): flag to load train or test
    """

    def __init__(self, root, train=True):
        super().__init__()
        # keeps parameters
        self.root = root
        self.train = train

        # creates paths
        config_path = os.path.join(root, 'config.json')
        if train:
            data_path = os.path.join(root, 'train.pt')
        else:
            data_path = os.path.join(root, 'test.pt')

        # loads config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # loads data
        self.data = torch.load(data_path)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        # selects params
        if self.train:
            mode = 'Train'
            size = self.config['trainsize']
        else:
            mode = 'Test'
            size = self.config['testsize']
        nb_sin = self.config['nb_sin']
        x_space = self.config['x_space']

        # prints to screen
        to_print = [f' {mode} SinMixDataset:']
        to_print.append(f'  size: {size}')
        to_print.append(f'  nb-sin: {nb_sin}')
        to_print.append(f'  x-space: {x_space}')

        return '\n'.join(to_print)
