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

if __name__ == '__main__':
    from common import printl, ERROR, INFO
else:
    from .common import printl, ERROR, INFO


"""
CLASSES
"""
class LineLayer(nn.Module):
    """
    Defines a Line function layer
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

        # computes the a
        y = a * x + b

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
        x_range = self.config['x_range']
        x_length = self.config['x_length']
        y_range = self.config['y_range']

        # prints to screen
        to_print = [f' {mode} LinesDataset:']
        to_print.append(f'  size: {size}')
        to_print.append(f'  x_range: {x_range}')
        to_print.append(f'  x_length: {x_length}')
        to_print.append(f'  y_range: {y_range}')

        return '\n'.join(to_print)


"""
FUNCTIONS
"""
def create(size, x_range, x_length, y_range):
    """
    Creates a Lines dataset
    ------------------------

    Args:
        size (int): the size of the dataset
        X_range (int): guarantees an output y inside [-x_range, x_range]
        x_length (int): the number of sampled x
        y_range (int): guarantees an output y inside [-y_range, y_range]

    Returns:
        List of List of FloatTensor: the dataset
    """
    # informs the user
    printl(INFO, f"""A new Lines dataset will be created with
          > x_range: {x_range}, 
          > x_length: {x_length}
          > y range: [-{y_range}, {y_range}]""")

    # computes new range for parameter a for y_range
    a_range = (2 * y_range) / (2 * x_range)

    # main creation loop
    data, list_params = [], []
    for i in range(size):
        # creates the x and y space
        x = np.linspace(-x_range, x_range, x_length)

        # draws parameter a
        a = 0
        while a == 0:
            a = random.random() * a_range * 2 - a_range

        # computes new range for parameter b
        if a > 0:
            b_min = -y_range + a * x_range
            b_max = y_range - a * x_range
        else:
            b_min = -y_range - a * x_range
            b_max = y_range + a * x_range

        # draws parameter b
        b = random.random() * (b_max - b_min) + b_min

        # computes the outputs
        y = a * x + b

        # checks if the y range works
        assert (np.abs(y) < y_range).all()

        # checks if the function already exists in the dataset
        if [a, b] in list_params:
            printl(INFO, 'The function is already chosen, skipping it')
        else:
            # adds the parameters to the list of parameters
            list_params.append([a, b])

            # converts to torch tensor
            y = torch.FloatTensor(y).float()
            params = torch.FloatTensor([a, b]).float()

            # adds the function to data
            data.append([y, params])

            # prints to screen
            if (i + 1) % int(0.1 * size) == 0:
                printl(INFO, 'Lines {:3}% done'.format(int(100*((i + 1) / size))))

    return data


if __name__ == '__main__':
    size = 25
    x_range = 10
    x_length = 1000
    y_range = 30
    h_range = 2 * x_range

    data = create(size, x_range, x_length, y_range)

    import matplotlib.pyplot as plt

    x = np.linspace(-x_range, x_range, x_length)
    _, axes = plt.subplots(5, 5)
    for i, d in enumerate(data):
        ax = axes[i//5][i%5]
        ax.plot(x, d[0].numpy())
        ax.set_xlim(-x_range, x_range)
        ax.set_ylim(-y_range, y_range)

    plt.tight_layout()
    plt.show()
