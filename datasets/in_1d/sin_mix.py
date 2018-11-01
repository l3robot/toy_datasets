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

from .common import printl, ERROR, INFO


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
        yy = self.amplitudes * torch.sin(self.phases * xx)
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
def create(size, nb_sin, x_min, x_max, x_length, amp_range, phase_range):
    """
    Creates a SinMix dataset
    ------------------------

    Args:
        size (int): the size of the dataset
        nb_sin (int): the number of sinus in a mix
        x_min (int): the min value of a sampled x
        x_max (int): the max value of a sampled x
        x_length (int): the number of sampled x
        amp_range (int): the range of amplitude [-amp_range, amp_range]
        phase_range (int): the range of phase [-phase_range, phase_range]
    """
    # informs the user
    printl(INFO, f"""A new SinMix dataset will be created with
          > {nb_sin} sinus
          > x_min: {x_min}, x_max {x_max}, x_length: {x_length}
          > amplitude range: [-{amp_range}, {amp_range}]
          > phase range: [-{phase_range}, {phase_range}]""")

    # main creation loop
    data, list_params = [], []
    for i in range(size):
        # creates the x and y space
        x = np.linspace(x_min, x_max, x_length)
        y = np.zeros_like(x)

        # creates random mixture of sin
        amps, phases = [], []
        for _ in range(nb_sin):
            # draw an amplitude and a phase
            amp = random.random() * amp_range * 2 - amp_range
            phase = random.random() * phase_range * 2 - phase_range

            # computes the outputs
            y += amp * np.sin(phase * x)

            # adds to the accumulators
            amps.append(amp)
            phases.append(phase)

        # checks if the function already exists in the dataset
        if [amps, phases] in list_params:
            printl(INFO, 'The function is already drawn, skipping it')
        else:
            # adds the parameters to the list of parameters
            list_params.append([amps, phases])

            # converts to torch tensor
            y = torch.FloatTensor(y).float()
            params = torch.FloatTensor([amps, phases]).float()

            # adds the function to data
            data.append([y, params])

            # prints to screen
            if (i + 1) % int(0.1 * size) == 0:
                printl(INFO, 'Mixture {:3}% done'.format(int(100*((i + 1) / size))))

    return data
