import numpy as np
import tensorflow_federated as tff


DATASET_MODULES = {'cifar100': tff.simulation.datasets.cifar100,
                   'emnist': tff.simulation.datasets.emnist,
                   'gldv2': tff.simulation.datasets.gldv2,
                   'experiments': tff.simulation.datasets.shakespeare,
                   'stackoverflow': tff.simulation.datasets.stackoverflow}


def extract_weights(client_datasets):
  weights = [len(list(client_dataset)) for client_dataset in client_datasets]
  weights = np.array(weights)
  return weights


def get_client_weights(name, limit_count=None):
  module = DATASET_MODULES[name]
  dataset = module.load_data()[0]
  client_datasets = dataset.datasets(limit_count=limit_count)
  weights = extract_weights(client_datasets)
  return weights
