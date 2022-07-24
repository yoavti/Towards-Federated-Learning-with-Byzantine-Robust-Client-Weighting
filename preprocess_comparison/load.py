import os

import tensorflow_federated as tff
import numpy as np


dataset_modules = {'cifar100': tff.simulation.datasets.cifar100,
                   'emnist': tff.simulation.datasets.emnist,
                   'gldv2': tff.simulation.datasets.gldv2,
                   'experiments': tff.simulation.datasets.shakespeare,
                   'stackoverflow': tff.simulation.datasets.stackoverflow}


def load_dataset(name):
    module = dataset_modules[name]
    cache_dir = os.path.join('preprocess_comparison', name, 'cache')
    return module.load_data(cache_dir=cache_dir)[0]


def get_client_weights(name, limit_count=None):
    dataset = load_dataset(name)
    client_datasets = dataset.datasets(limit_count=limit_count)
    weights = [len(list(client_dataset)) for client_dataset in client_datasets]
    weights = np.array(weights)
    return weights
