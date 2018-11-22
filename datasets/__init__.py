"""This module defines functions and classes related to datasets
"""
import os
import json
import pkgutil


HERE = os.path.dirname(os.path.realpath(__file__))

IN_1D_PATH = os.path.join(HERE, 'in_1d')
IN_2D_PATH = os.path.join(HERE, 'in_2d')

DATASETS_LIST = {m.name: m.module_finder.path for m in
                 pkgutil.iter_modules([IN_1D_PATH, IN_2D_PATH]) if m != 'common'}


def get_dataset_class(dataset_type):
    """Gets the right dataset class for a dataset type

    Args:
        dataset_type (str): type of the dataset

    Returns:
        DatasetClass: the right class

    Raises:
        ValueError: if dataset_type is not a valid dataset
    """
    dataset_type = dataset_type.replace('-', '_')

    if dataset_type in DATASETS_LIST:
        dim_type = os.path.split(DATASETS_LIST[dataset_type])[-1]
        top = getattr(__import__(f'{dim_type}.{dataset_type}'), dataset_type)
        class_name = [c for c in dir(top) if 'Dataset' in c][1]
        DatasetClass = getattr(top, class_name)
    else:
        print(f'  [!] Error: {dataset_type} is not a valid dataset type')
        raise ValueError

    return DatasetClass


def load_dataset_config(dataset_path):
    """Loads a dataset config

    Args:
        dataset_type (str): type of the dataset

    Returns:
        dict: the dataset config
    """
    config_path = os.path.join(dataset_path, 'config.json')
    with open(config_path, 'r') as f:
        dataset_config = json.load(f)
    return dataset_config


def load_dataset(dataset_type, dataset_path, train=False):
    """Loads a dataset

    Args:
        dataset_type (str): type of the dataset
        dataset_path (str): path of the dataset
        train (bool): flag to load the training dataset
                      instead of the test dataset

    Returns:
        DatasetClass: the dataset with the right class
    """
    DatasetClass = get_dataset_class(dataset_type)
    dataset = DatasetClass(dataset_path, train=train)
    return dataset
