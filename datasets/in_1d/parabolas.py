"""
This module defines the lines dataset
-------------------------------------

author: Louis-Émile Robitaille (l3robot)
date-of-creation: 2018-11-01
"""

import os
import json
import random

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .common import printl, ERROR, INFO


"""
CLASSES
"""
class ParabolaLayer(nn.Module):
    """
    Defines a Parabola function layer
    -------------------------------

    Args:
        z_size (FloatTensor): size of the function representation
    """

    def __init__(self, z_size):
        super().__init__()
        # keeps parameters
        self.z_size = z_size

        # creates layers
        self.a = nn.Linear(z_size, 1)
        self.b = nn.Linear(z_size, 1)
        self.c = nn.Linear(z_size, 1)

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
        a = self.a(z)
        b = self.b(z)
        c = self.c(z)

        # computes the slope
        y = a * x ** 2 + b * x + c

        return y


class ParabolasDataset(Dataset):
    """
    Defines the ParabolasDataset
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
        x_min = self.config['x_min']
        x_max = self.config['x_max']
        x_length = self.config['x_length']
        a_range = self.config['a_range']
        b_range = self.config['b_range']
        c_range = self.config['c_range']

        # prints to screen
        to_print = [f' {mode} LinesDataset:']
        to_print.append(f'  size: {size}')
        to_print.append(f'  x_min: {x_min}')
        to_print.append(f'  x_max: {x_max}')
        to_print.append(f'  x_length: {x_length}')
        to_print.append(f'  a_range: {a_range}')
        to_print.append(f'  b_range: {b_range}')
        to_print.append(f'  c_range: {c_range}')

        return '\n'.join(to_print)


"""
FUNCTIONS
"""
def create(size, x_min, x_max, x_length, a_range, b_range, c_range):
    """
    Creates a Parabolas dataset
    ------------------------

    Args:
        size (int): the size of the dataset
        x_min (int): the min value of a sampled x
        x_max (int): the max value of a sampled x
        x_length (int): the number of sampled x
        a_range (int): the range of slope [-a_range, a_range]
        b_range (int): the range of y-intercept [-b_range, b_range]
        c_range (int): the range of y-intercept [-c_range, c_range]

    Returns:
        List of List of FloatTensor: the dataset
    """
    # informs the user
    printl(INFO, f"""A new Parabolas dataset will be created with
          > x_min: {x_min}, x_max {x_max}, x_length: {x_length}
          > a range: [-{a_range}, {a_range}]
          > b range: [-{b_range}, {b_range}]
          > c range: [-{c_range}, {c_range}]""")

    # main creation loop
    data, list_params = [], []
    for i in range(size):
        # creates the x and y space
        x = np.linspace(x_min, x_max, x_length)
        y = np.zeros_like(x)

        # draw an amplitude and a phase
        a = random.random() * a_range * 2 - a_range
        b = random.random() * b_range * 2 - b_range
        c = random.random() * c_range * 2 - c_range

        # computes the outputs
        y += a * x ** 2 + b * x + c

        # checks if the function already exists in the dataset
        if [a, b, c] in list_params:
            printl(INFO, 'The function is already chosen, skipping it')
        else:
            # adds the parameters to the list of parameters
            list_params.append([a, b, c])

            # converts to torch tensor
            y = torch.FloatTensor(y).float()
            params = torch.FloatTensor([a, b, c]).float()

            # adds the function to data
            data.append([y, params])

            # prints to screen
            if (i + 1) % int(0.1 * size) == 0:
                printl(INFO, 'Parabolas {:3}% done'.format(int(100*((i + 1) / size))))

    return data
