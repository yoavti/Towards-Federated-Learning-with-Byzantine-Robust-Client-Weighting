import numpy as np
from matplotlib import pyplot as plt


def l1_metric(v1, v2):
    return np.linalg.norm(v1 - v2, 1)


def plot_weights(named_weights):
    weights_list = named_weights.values()
    weights_list = list(weights_list)
    x = np.arange(len(weights_list[0]))
    fig, axs = plt.subplots(1, len(named_weights), sharex='all', sharey='all')
    for (name, weights), ax in zip(named_weights.items(), axs):
        ax.title.set_text(name)
        ax.bar(x, weights)
    plt.show()
