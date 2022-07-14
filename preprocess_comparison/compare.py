from absl import app, flags

from preprocess_comparison.comparison_utils import plot_weights, available_metrics
from preprocess_comparison.load import dataset_modules, get_client_weights
from shared.truncate import truncate
from shared.lp import lp
from shared.utils import is_valid_solution, maximal_weight_proportion
from utils import utils_impl

from pprint import PrettyPrinter

pp = PrettyPrinter()

available_preprocess = {'truncate': truncate, 'lp': lp}


with utils_impl.record_hparam_flags() as comparison_flags:
    flags.DEFINE_enum('dataset', 'emnist', list(dataset_modules), 'Which dataset to take weights from.')
    flags.DEFINE_integer('limit_count', None, 'Number of weights to take from dataset.')
    flags.DEFINE_multi_enum('weight_preproc', list(available_preprocess), list(available_preprocess),
                            'What to do with the clients\' relative weights.')
    flags.DEFINE_float('alpha', 0.1, 'Byzantine proportion.')
    flags.DEFINE_float('alpha_star', 0.5, 'Byzantine weight proportion.')
    flags.DEFINE_multi_enum('metrics', list(available_metrics), list(available_metrics),
                            'How to compare the results of preprocess to the original weights.')
    flags.DEFINE_bool('check_validity', True, 'Whether to check if a given preprocess solution gives a valid mwp.')
    flags.DEFINE_bool('plot', True, 'Whether to plot the different weights.')
    flags.DEFINE_bool('compare_mwp', True, 'Whether to print the mwp of each preprocess output.')

FLAGS = flags.FLAGS


def compare_weights(original_weights, named_new_weights, named_metrics,
                    alpha=0.1, alpha_star=0.5,
                    check_validity=True, plot=True, compare_mwp=True):
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


def main(_):
    # loading client weights
    weights = get_client_weights(FLAGS.dataset, FLAGS.limit_count)
    # applying different preprocess procedures
    selected_preprocess = {name: available_preprocess[name] for name in FLAGS.weights_preproc}
    named_new_weights = {name: preprocess(weights, alpha=FLAGS.alpha, alpha_star=FLAGS.alpha_star)
                         for name, preprocess in selected_preprocess}
    # comparing the outputs
    selected_metrics = {name: available_metrics[name] for name in FLAGS.metrics}
    compare_weights(weights, named_new_weights, selected_metrics,
                    FLAGS.alpha, FLAGS.alpha_star, FLAGS.check_validity, FLAGS.plot, FLAGS.compare_mwp)


if __name__ == '__main__':
  app.run(main)
