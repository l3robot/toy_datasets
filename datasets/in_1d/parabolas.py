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
        x_range = self.config['x_range']
        x_length = self.config['x_length']
        y_range = self.config['y_range']
        h_range = self.config['a_range']

        # prints to screen
        to_print = [f' {mode} LinesDataset:']
        to_print.append(f'  size: {size}')
        to_print.append(f'  x_range: {x_range}')
        to_print.append(f'  x_length: {x_length}')
        to_print.append(f'  y_range: {y_range}')
        to_print.append(f'  h_range: {h_range}')

        return '\n'.join(to_print)


"""
FUNCTIONS
"""
def create(size, x_range, x_length, y_range, h_range):
    """
    Creates a Parabolas dataset
    ------------------------

    Args:
        size (int): the size of the dataset
        x_range (int): guarantees an output x inside [-x_range, x_range]
        x_length (int): the number of sampled x
        y_range (int): guarantees an output y inside [-y_range, y_range]
        h_range (int): the range of the maxima's x [-h_range, h_range]

    Returns:
        List of List of FloatTensor: the dataset
    """
    # informs the user
    printl(INFO, f"""A new Parabolas dataset will be created with
          > x_range: [-{x_range}, {x_range}], 
          > x_length: {x_length}
          > y range: [-{y_range}, {y_range}]
          > h range: [-{h_range}, {h_range}]""")

    # main creation loop
    data, list_params = [], []
    for i in range(size):
        # creates the x and y space
        x = np.linspace(-x_range, x_range, x_length)

        # draws parameter h
        h = random.random() * h_range * 2 - h_range

        # computes repetitive quantities
        pre_min = (x_range + h) ** 2
        pre_max = (x_range - h) ** 2

        # draws parameter a and k
        if abs(h) > x_range:
            # computes range of parameter a
            a_range = 2 * y_range / (pre_max - pre_min)
            a_min = -a_range
            a_max = a_range

            # draws parameter a
            a = 0
            while a == 0:
                a = random.random() * (a_max - a_min) + a_min

            # computes range for parameter k
            if h < -x_range and a > 0 or h > x_range and a < 0:
                k_min = -y_range - a * pre_min
                k_max = y_range - a * pre_max
            else:
                k_min = -y_range - a * pre_max
                k_max = y_range - a * pre_min
        else:
            # computes range of parameter a
            if h < 0:
                a_range = 2 * y_range / pre_max
            else:
                a_range = 2 * y_range / pre_min
            a_min = -a_range
            a_max = a_range

            # draws parameter a
            a = random.random() * (a_max - a_min) + a_min

            # computes range for parameter k
            if h < 0 and a > 0:
                k_min = -y_range
                k_max = y_range - a * pre_max
            elif h > 0 and a < 0:
                k_min = -y_range - a * pre_min
                k_max = y_range
            elif h < 0 and a < 0:
                k_min = -y_range - a * pre_max
                k_max = y_range
            else:
                k_min = -y_range
                k_max = y_range - a * pre_min

        # draws parameter k
        k = random.random() * (k_max - k_min) + k_min

        # convert to a, b, c
        b = -2 * a * h
        c = a * (h ** 2) + k

        # computes the outputs
        y = a * x ** 2 + b * x + c

        # checks if the y range works
        try:
            assert (np.abs(y) <= y_range).all()
        except AssertionError:
            print('-> {:.2f}'.format(2 * y_range / (pre_max - pre_min)))
            import matplotlib.pyplot as plt
            _, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(x, y)
            ax2.plot(x, a * (x - h) ** 2 + k)
            plt.show()
            raise AssertionError

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


if __name__ == '__main__':
    size = 25
    x_range = 10
    x_length = 1000
    y_range = 30
    h_range = 2 * x_range

    data = create(size, x_range, x_length, y_range, h_range)

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
