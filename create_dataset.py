"""
This script can be used to create a dataset

author: Louis-Émile Robitaille (l3robot)
date-of-creation: 2018-11-01
"""

import os
import time
import json
import random

import click

import torch

from common import printl, ERROR, INFO


def create_dataset(dataset_type, root, train_ratio, create_fct, **config):
    """
    Creates an arbitrary dataset
    ----------------------------

    Args:
        dataset_type (str): type of the selected dataset
        root (str): path where to save the dataset
        train_ratio (float): ratio of the train dataset
        create_fct (function): function of the selected dataset
        config (dict): configuration of the selected dataset
    """
    # check if the dataset folder already exists
    if os.path.isdir(root):
        printl(ERROR, 'A dataset already exists at this location')
        exit()

    # creates the datasets
    printl(INFO, 'Creating the dataset')
    dataset = create_fct(**config)

    # splits the dataset in train and test
    train_size = int(train_ratio * len(dataset))
    train_idx = random.sample(range(len(dataset)), k=train_size)
    train_set = [dataset[i] for i in range(len(dataset)) if i in train_idx]
    test_set = [dataset[i] for i in range(len(dataset)) if i not in train_idx]

    # creates the root folder
    os.mkdir(root)

    # creates the paths
    config_path = os.path.join(root, 'config.json')
    train_path = os.path.join(root, 'train.pt')
    test_path = os.path.join(root, 'test.pt')

    # save the config
    config['type'] = dataset_type
    config['train_size'] = train_size
    config['test_size'] = len(dataset) - train_size
    config['timestamp'] = time.time()
    with open(config_path, 'w') as f:
        json.dump(config, f)

    # save the dataset
    torch.save(train_set, train_path)
    torch.save(test_set, test_path)


@click.group()
def datasets():
    """
    This cli can be used to create a dataset
    """
    pass


@click.command()
@click.argument('root')
@click.option('--train-ratio', default=0.1, help='size of the dataset')
@click.option('--size', default=60000, help='size of the dataset')
@click.option('--nb-sin', default=1, help='the number of sinus in a mix')
@click.option('--x-range', default=5, help='the guaranteed range of output x [-x_range, x_range]')
@click.option('--x-length', default=1000, help='the number of sampled x')
@click.option('--y-range', default=30, help='the guaranteed range of output y [-y_range, y_range]')
@click.option('--f-range', default=5, help='the range of f [-f_range, f_range]')
def sin_mix(root, train_ratio, **config):
    """
    Creates a sin-mix dataset

    Args:
        root (str): path where to save the dataset
        train-ratio (float): ratio of the train dataset
        size (int): the size of the dataset (default: 6000)
        nb_sin (int): the number of sinus in a mix (default: 3)
        x_range (int): the guaranteed range of output x [-x_range, x_range] (default=5)
        x_length (int): the number of sampled x (default: 1000)
        y_range (int): the guaranteed range of output y [-y_range, y_range] (default=30)
        f_range (int): the range of f [-f_range, f_range] (default: 5)
    """
    from datasets.in_1d.sin_mix import create
    create_dataset("sin-mix", root, train_ratio, create, **config)


@click.command()
@click.argument('root')
@click.option('--train-ratio', default=0.1, help='size of the dataset')
@click.option('--size', default=60000, help='size of the dataset')
@click.option('--x-range', default=5, help='the guaranteed range of output x [-x_range, x_range]')
@click.option('--x-length', default=1000, help='the number of sampled x')
@click.option('--y-range', default=30, help='the guaranteed range of output y [-y_range, y_range]')
def lines(root, train_ratio, **config):
    """
    Creates a sin-mix dataset

    Args:
        root (str): path where to save the dataset
        train-ratio (float): ratio of the train dataset
        size (int): the size of the dataset (default: 6000)
        x_range (int): the guaranteed range of output x [-x_range, x_range] (default=5)
        x_length (int): the number of sampled x (default: 1000)
        y_range (int): the guaranteed range of output y [-y_range, y_range] (default=30)
    """
    from datasets.in_1d.lines import create
    create_dataset("lines", root, train_ratio, create, **config)


@click.command()
@click.argument('root')
@click.option('--train-ratio', default=0.1, help='size of the dataset')
@click.option('--size', default=60000, help='size of the dataset')
@click.option('--x-range', default=5, help='the guaranteed range of output x [-x_range, x_range]')
@click.option('--x-length', default=1000, help='the number of sampled x')
@click.option('--y-range', default=30, help='the guaranteed range of output y [-y_range, y_range]')
@click.option('--h-range', default=4, help='the range of parameter h [-h_range, h_range]')
def parabolas(root, train_ratio, **config):
    """
    Creates a sin-mix dataset

    Args:
        root (str): path where to save the dataset
        train-ratio (float): ratio of the train dataset
        size (int): the size of the dataset (default: 6000)
        x_range (int): the guaranteed range of output x [-x_range, x_range] (default=5)
        x_length (int): the number of sampled x (default: 1000)
        y_range (int): the guaranteed range of output y [-y_range, y_range] (default=30)
        h_range (int): the range of parameter h [-h_range, h_range] (default: 4)
    """
    from datasets.in_1d.parabolas import create
    create_dataset("parabolas", root, train_ratio, create, **config)


datasets.add_command(sin_mix)
datasets.add_command(lines)
datasets.add_command(parabolas)


if __name__ == '__main__':
    datasets()
