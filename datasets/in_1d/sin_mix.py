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

if __name__ == '__main__':
    from common import printl, ERROR, INFO
else:
    from .common import printl, ERROR, INFO


"""
CLASSES
"""
class SinMixLayer(nn.Module):
    """
    Defines a SinMix function layer
    -------------------------------

    y += a * np.sin(f * x + p)

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
        self.a = nn.Linear(z_size, nb_sin)
        self.f = nn.Linear(z_size, nb_sin)
        self.p = nn.Linear(z_size, nb_sin)

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
        f = self.f(z)
        p = self.p(z)

        # expand x for one-time computation
        xx = x.expand(-1, self.nb_sin)

        # computes all sinus and mixes them
        yy = a * torch.sin(f * xx + p)
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
        x_range = self.config['x_range']
        x_length = self.config['x_length']
        y_range = self.config['y_range']
        f_range = self.config['f_range']

        # prints to screen
        to_print = [f' {mode} SinMixDataset:']
        to_print.append(f'  size: {size}')
        to_print.append(f'  nb_sin: {nb_sin}')
        to_print.append(f'  x_range: {x_range}')
        to_print.append(f'  x_length: {x_length}')
        to_print.append(f'  y_range: {y_range}')
        to_print.append(f'  f_range: {f_range}')

        return '\n'.join(to_print)


"""
FUNCTIONS
"""
def create(size, nb_sin, x_range, x_length, y_range, f_range):
    """
    Creates a SinMix dataset
    ------------------------

    Args:
        size (int): the size of the dataset
        nb_sin (int): the number of sinus in a mix
        y_range (int): guarantees an output x inside [-x_range, x_range]
        x_length (int): the number of sampled x
        y_range (int): guarantees an output y inside [-y_range, y_range]
        f_range (int): the range of f [-a_range, a_range]

    Returns:
        List of List of FloatTensor: the dataset
    """
    # informs the user
    printl(INFO, f"""A new SinMix dataset will be created with
          > {nb_sin} sinus
          > x_min: {x_range}, 
          > x_length: {x_length}
          > y range: [-{y_range}, {y_range}]
          > f range: [0, {f_range}]""")

    # computes a new range
    a_range = y_range / nb_sin

    # main creation loop
    data, list_params = [], []
    for i in range(size):
        # creates the x and y space
        x = np.linspace(-x_range, x_range, x_length)
        y = np.zeros_like(x)

        # creates random mixture of sin
        aa, ff, pp = [], [], []
        for _ in range(nb_sin):
            # draws parameter a
            a = 0
            while a == 0:
                a = random.random() * a_range * 2 - a_range

            # draws parameter f
            f = 0
            while f == 0:
                f = random.random() * f_range
            
            # draws parameter p
            p = random.random() * 2 * np.pi

            # computes the outputs
            y += a * np.sin(f * x + p)

            # adds to the accumulators
            aa.append(a)
            ff.append(f)
            pp.append(p)

        # checks if the y range works
        assert (np.abs(y) < y_range).all()

        # checks if the function already exists in the dataset
        if [aa, ff, pp] in list_params:
            printl(INFO, 'The function is already chosen, skipping it')
        else:
            # adds the parameters to the list of parameters
            list_params.append([aa, ff, pp])

            # converts to torch tensor
            y = torch.FloatTensor(y).float()
            params = torch.FloatTensor([aa, ff, pp]).float()

            # adds the function to data
            data.append([y, params])

            # prints to screen
            if (i + 1) % int(0.1 * size) == 0:
                printl(INFO, 'Mixture {:3}% done'.format(int(100*((i + 1) / size))))

    return data


if __name__ == '__main__':
    size = 25
    nb_sin = 1
    x_range = 10
    x_length = 1000
    y_range = 30
    f_range = 5

    data = create(size, nb_sin, x_range, x_length, y_range, f_range)

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
