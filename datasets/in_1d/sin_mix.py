"""
This script can be used to create a SinMixDataSet
-------------------------------------------------

author: Louis-Émile Robitaille (l3robot)
date-of-creation: 2018-10-11
"""

import os
import time
import json
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import printl, ERROR, WARNING, INFO 


"""
CONFIG
"""


SIZE = 6000
NB_SIN = 3

X_MIN = 0
X_MAX = 10
X_LENGTH = 1000

AMP_RANGE = 4
PHASE_RANGE = 5


"""
CLASSES
"""


class SinMixLayer(nn.Module):
    """
    Defines a SinMix function layer
    -------------------------------

    Args:
        amplitudes (int or FloatTensor): amplitudes of the sinus
        phases (int or FloatTensor): phases of the sinus
    """

    def __init__(self, amplitudes, phases):
        super().__init__()
        # keeps parameters
        self.amplitudes = nn.Paramater(amplitudes)
        self.phases = nn.Paramater(phases)

        # computes the number of sinus
        self.nb_sin = len(amplitudes)

    def forward(self, x):
        """
        Forward pass
        ------------

        Args:
            x (FloatTensor): input of the function
        """
        # checks the size of the tensor
        if len(x.size()) != 2:
            printl(ERROR, f'x must have 2 axis, now {len(x.size())}')
            raise AttributeError

        # checks if the data is one dimensional
        if x.size()[-1] != 1:
            printl(ERROR, f'x must have 1 dimensions, now {x.size()[-1]}')
            raise AttributeError

        # expand x for one-time computation
        xx = x.expand(-1, self.nb_sin)

        # computes all sinus and mixes them
        yy = amplitudes * torch.sin(phases * xx)
        yy = yy.sum(dim=1)

        return yy


class SinMixDataset(Dataset):
    """
    Defines the SinMixDataset
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


"""
FUNCTIONS
"""


def create_sin_mix(size=SIZE, nb_sin=NB_SIN):
    """
    TODO: docstring
    """
    data = []
    for i in range(size):
        # creates the x and y space
        x = np.linspace(X_MIN, X_MAX, X_LENGTH)
        y = np.zeros_like(x)

        # creates random mixture of sin
        des, amps, phases = [], [], []
        for _ in range(nb_sin):
            amp = random.random() * AMP_RANGE * 2 - AMP_RANGE
            phase = random.random() * PHASE_RANGE * 2 - PHASE_RANGE
            y += amp * np.sin(phase * x)
            amps.append(amp); phases.append(phase)
            des.append(f'{amp:.2f}sin({phase:.2f}x)')

        # convert to torch tensor
        y = torch.tensor(y).float()
        params = torch.tensor([amps, phases]).float()

        # adds the sin to data
        data.append([y, params])

        # print to screen
        if (i + 1) % int(0.1 * size) == 0:
            print(' [-] Mixture {:3}% done'.format(int(100*((i + 1) / size))))

    return data


if __name__ == '__main__':
    # creates args parser
    parser = argparse.ArgumentParser(description='Creating a SinMixDataset')

    # defines dataset arguments
    parser.add_argument('--trainsize', metavar='size', type=int,\
        help='number of sin mixtures in trainset', default=SIZE)
    parser.add_argument('--testsize', metavar='size', type=int,\
        help='number of sin mixtures in testset', default=SIZE)
    parser.add_argument('--nb-sin', metavar='nb', type=int,\
        help='number of sin in a mixture (2 params per sin)', default=NB_SIN)

    # defines logistic arguments
    parser.add_argument('root', metavar='folder', type=str,
                        help='location of the dataset root folder')

    # parses args
    config = vars(parser.parse_args())

    # creates the datasets
    print(' [-] Creating the train dataset')
    trainset = create_sin_mix(config['trainsize'], config['nb_sin'])
    print(' [-] Creating the test dataset')
    testset = create_sin_mix(config['testsize'], config['nb_sin'])

    # check if the dataset folder already exists
    if os.path.isdir(config['root']):
        print(' [-] A dataset already exists at this location')
        exit()

    else:
        # creates the root folder
        os.mkdir(config['root'])

        # creates the paths
        config_path = os.path.join(config['root'], 'config.json')
        train_path = os.path.join(config['root'], 'train.pt')
        test_path = os.path.join(config['root'], 'test.pt')

        # save the config
        config['x_space'] = [X_MIN, X_MAX, X_LENGTH]
        config['timestamp'] = time.time()
        with open(config_path, 'w') as f:
            json.dump(config, f)

        # save the dataset
        torch.save(trainset, train_path)
        torch.save(testset, test_path)
