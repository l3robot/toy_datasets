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
class LinesLayer(nn.Module):
    """
    Defines a Lines function layer
    -------------------------------

    Args:
        slope (float or FloatTensor): slope of the line
        y_intercept (float or FloatTensor): y-intercept of the line
    """

    def __init__(self, slope, y_intercept):
        super().__init__()
        # keeps parameters
        self.slope = nn.Paramater(slope)
        self.y_intercept = nn.Paramater(y_intercept)

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

        # computes the slope
        y = self.slope * x + self.y_intercept

        return y


class LinesDataset(Dataset):
    """
    Defines the LinesDataset
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
        slope_range = self.config['slope_range']
        y_intercept_range = self.config['y_intercept_range']

        # prints to screen
        to_print = [f' {mode} LinesDataset:']
        to_print.append(f'  size: {size}')
        to_print.append(f'  x_min: {x_min}')
        to_print.append(f'  x_max: {x_max}')
        to_print.append(f'  x_length: {x_length}')
        to_print.append(f'  slope_range: {slope_range}')
        to_print.append(f'  y_intercept_range: {y_intercept_range}')

        return '\n'.join(to_print)


"""
FUNCTIONS
"""
def create(size, x_min, x_max, x_length, slope_range, y_intercept_range):
    """
    Creates a Lines dataset
    ------------------------

    Args:
        size (int): the size of the dataset
        x_min (int): the min value of a sampled x
        x_max (int): the max value of a sampled x
        x_length (int): the number of sampled x
        slope_range (int): the range of slope [-slope_range, slope_range]
        y_intercept_range (int): the range of y-intercept [-y_intercept_range, y_intercept_range]
    """
    # informs the user
    printl(INFO, f"""A new Lines dataset will be created with
          > x_min: {x_min}, x_max {x_max}, x_length: {x_length}
          > slope range: [-{slope_range}, {slope_range}]
          > y-intercept range: [-{y_intercept_range}, {y_intercept_range}]""")

    # main creation loop
    data, list_params = [], []
    for i in range(size):
        # creates the x and y space
        x = np.linspace(x_min, x_max, x_length)
        y = np.zeros_like(x)

        # draw an amplitude and a phase
        slope = random.random() * slope_range * 2 - slope_range
        y_intercept = random.random() * y_intercept_range * 2 - y_intercept_range

        # computes the outputs
        y += slope * x + y_intercept

        # checks if the function already exists in the dataset
        if [slope, y_intercept] in list_params:
            printl(INFO, 'The function is already chosen, skipping it')
        else:
            # adds the parameters to the list of parameters
            list_params.append([slope, y_intercept])

            # converts to torch tensor
            y = torch.FloatTensor(y).float()
            params = torch.FloatTensor([slope, y_intercept]).float()

            # adds the function to data
            data.append([y, params])

            # prints to screen
            if (i + 1) % int(0.1 * size) == 0:
                printl(INFO, 'Lines {:3}% done'.format(int(100*((i + 1) / size))))

    return data
