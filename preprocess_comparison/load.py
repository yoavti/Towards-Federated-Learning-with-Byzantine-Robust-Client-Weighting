import os

import tensorflow_federated as tff


dataset_modules = {'cifar100': tff.simulation.datasets.cifar100,
                   'emnist': tff.simulation.datasets.emnist,
                   'gldv2': tff.simulation.datasets.gldv2,
                   'shakespeare': tff.simulation.datasets.shakespeare,
                   'stackoverflow': tff.simulation.datasets.stackoverflow}


def load_dataset(name):
    module = dataset_modules[name]
    cache_dir = os.path.join('preprocess_comparison', name, 'cache')
    return module.load_data(cache_dir=cache_dir)[0]
