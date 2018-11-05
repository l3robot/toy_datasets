"""
This module defines the SinMixDataSet dataset
---------------------------------------------

author: Louis-Émile Robitaille (l3robot)
date-of-creation: 2018-10-11
"""

import os
import json
import random

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .common import printl, ERROR, WARNING, INFO


"""
CLASSES
"""
class SinMixLayer(nn.Module):
    """
    Defines a SinMix function layer
    -------------------------------

    y += a * np.sin(b * x + c)

    Args:
        z_size (FloatTensor): size of the function representation
        nb_sin (int): number of sinus in the mix
    """

    def __init__(self, z_size, nb_sin):
        super().__init__()
        # keeps parameters
        self.z_size = z_size
        self.nb_sin = nb_sin

        # creates layers
        self.aa = nn.Linear(z_size, nb_sin)
        self.bb = nn.Linear(z_size, nb_sin)
        self.cc = nn.Linear(z_size, nb_sin)

    def forward(self, x, z):
        """
        Forward pass
        ------------

        Args:
            x (FloatTensor): input of the function
            z (FloatTensor): representation of the function

        Returns:
            FloatTensor: the output of the function
        """
        # checks the size of the tensor
        if len(x.size()) != 2:
            printl(ERROR, f'x must have 2 axis, now {len(x.size())}')
            raise AttributeError

        # checks if the data is one dimensional
        if x.size()[-1] != 1:
            printl(ERROR, f'x must have 1 dimensions, now {x.size()[-1]}')
            raise AttributeError

        # computes the parameters of the function
        aa = self.aa(z)
        bb = self.bb(z)
        cc = self.cc(z)

        # expand x for one-time computation
        xx = x.expand(-1, self.nb_sin)

        # computes all sinus and mixes them
        yy = aa * torch.sin(bb * xx + cc)
        y = yy.sum(dim=1)

        return y


class SinMixDataset(Dataset):
    """
    Defines the SinMixDataset
    -------------------------

    Args:
        root (str): location of the dataset
        train (bool): flag to load train or test (default=True)
    """

    def __init__(self, root, train=True):
        super().__init__()
        # keeps parameters
        self.root = root
        self.train = train

        # creates paths
        self.config_path = os.path.join(root, 'config.json')
        if train:
            self.data_path = os.path.join(root, 'train.pt')
        else:
            self.data_path = os.path.join(root, 'test.pt')

        # loads config
        with open(self.config_path, 'r') as file:
            self.config = json.load(file)

        # loads data
        self.data = torch.load(self.data_path)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        # selects params
        if self.train:
            mode = 'Train'
            size = self.config['train_size']
        else:
            mode = 'Test'
            size = self.config['test_size']

        # parses the config
        nb_sin = self.config['nb_sin']
        x_min = self.config['x_min']
        x_max = self.config['x_max']
        x_length = self.config['x_length']
        y_range = self.config['y_range']
        a_range = self.config['a_range']
        b_range = self.config['b_range']
        c_range = self.config['c_range']

        # prints to screen
        to_print = [f' {mode} SinMixDataset:']
        to_print.append(f'  size: {size}')
        to_print.append(f'  nb_sin: {nb_sin}')
        to_print.append(f'  x_min: {x_min}')
        to_print.append(f'  x_max: {x_max}')
        to_print.append(f'  x_length: {x_length}')
        to_print.append(f'  y_range: {y_range}')
        to_print.append(f'  a_range: {a_range}')
        to_print.append(f'  b_range: {b_range}')
        to_print.append(f'  c_range: {c_range}')

        return '\n'.join(to_print)


"""
FUNCTIONS
"""
def create(size, nb_sin, x_min, x_max, x_length, y_range, a_range, b_range, c_range):
    """
    Creates a SinMix dataset
    ------------------------

    Args:
        size (int): the size of the dataset
        nb_sin (int): the number of sinus in a mix
        x_min (int): the min value of a sampled x
        x_max (int): the max value of a sampled x
        x_length (int): the number of sampled x
        y_range (int): guarantees an output y inside [-y_range, y_range]
        a_range (int): the range of a [-a_range, a_range]
        b_range (int): the range of b [-b_range, b_range]
        c_range (int): the range of c [-c_range, c_range]

    Returns:
        List of List of FloatTensor: the dataset
    """
    # informs the user
    printl(INFO, f"""A new SinMix dataset will be created with
          > {nb_sin} sinus
          > x_min: {x_min}, x_max {x_max}, x_length: {x_length}
          > y range: [-{y_range}, {y_range}]
          > a range: [-{a_range}, {a_range}]
          > b range: [-{b_range}, {b_range}]
          > c range: [-{c_range}, {c_range}]""")

    # computes a new range
    a_new_range = y_range / nb_sin
    a_new_range = min(a_new_range, a_range)

    # informs the user
    if a_new_range != a_range:
        printl(WARNING, f"""The range of parameter a [-{a_range:.2f}, {a_range:.2f}]
            has been change to [-{a_new_range:.2f}, {a_new_range:.2f}]
            due to the range of y [-{y_range:.2f}, {y_range:.2f}]""")

    # main creation loop
    data, list_params = [], []
    for i in range(size):
        # creates the x and y space
        x = np.linspace(x_min, x_max, x_length)
        y = np.zeros_like(x)

        # creates random mixture of sin
        aa, bb, cc = [], [], []
        for _ in range(nb_sin):
            # draw an a and a b
            a = random.random() * a_new_range * 2 - a_new_range
            b = random.random() * b_range * 2 - b_range
            c = random.random() * c_range * 2 - c_range

            # computes the outputs
            y += a * np.sin(b * x + c)

            # adds to the accumulators
            aa.append(a)
            bb.append(b)
            cc.append(c)

        # checks if the y range works
        assert (np.abs(y) < y_range).all()

        # checks if the function already exists in the dataset
        if [aa, bb] in list_params:
            printl(INFO, 'The function is already chosen, skipping it')
        else:
            # adds the parameters to the list of parameters
            list_params.append([aa, bb, cc])

            # converts to torch tensor
            y = torch.FloatTensor(y).float()
            params = torch.FloatTensor([aa, bb, cc]).float()

            # adds the function to data
            data.append([y, params])

            # prints to screen
            if (i + 1) % int(0.1 * size) == 0:
                printl(INFO, 'Mixture {:3}% done'.format(int(100*((i + 1) / size))))

    return data
