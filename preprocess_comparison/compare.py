import numpy as np
import matplotlib.pyplot as plt

from shared.utils import is_valid_solution, maximal_weight_proportion

from pprint import PrettyPrinter

pp = PrettyPrinter()


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


def compare_weights(original_weights, named_new_weights, named_metrics,
                    alpha, alpha_star,
                    check_validity, plot, compare_mwp):
    # checking solution validity
    if check_validity:
        for preprocess_name, new_weights in named_new_weights.items():
            if not is_valid_solution(new_weights, alpha, alpha_star):
                mwp = maximal_weight_proportion(new_weights, alpha)
                print(f'{preprocess_name} did not produce a valid solution. mwp={mwp}, whereas alpha_star={alpha_star}')
    # plotting solutions
    if plot:
        plot_weights(named_new_weights | {'original': original_weights})
    # comparing mwp
    if compare_mwp:
        print('mwp')
        pp.pprint({preprocess_name: maximal_weight_proportion(new_weights, alpha)
                   for preprocess_name, new_weights in named_new_weights})
    # compare metrics
    if named_metrics:
        print('metrics')
        for metric_name, metric in named_metrics.items():
            print(metric_name)
            pp.pprint({preprocess_name: metric(new_weights, original_weights)
                       for preprocess_name, new_weights in named_new_weights.items()})
