import numpy as np
from absl import app, flags
from pprint import PrettyPrinter

from experiments.preprocess_comparison.utils.comparison_utils import plot_weights, available_metrics
from shared.extract_client_weights import DATASET_MODULES, get_client_weights

from shared.google_tff_research.utils import utils_impl
from shared.preprocess import PREPROC_TRANSFORMS
from shared.preprocess.utils import is_valid_solution, maximal_weight_proportion
from shared.flags_validators import create_optional_validator, check_positive, check_proportion

pp = PrettyPrinter()


with utils_impl.record_hparam_flags() as comparison_flags:
  flags.DEFINE_enum('dataset', 'emnist', list(DATASET_MODULES), 'Which dataset to take weights from.')
  flags.DEFINE_integer('limit_count', None, 'Number of weights to take from dataset.')
  flags.DEFINE_multi_enum('preprocess_funcs', list(PREPROC_TRANSFORMS), list(PREPROC_TRANSFORMS),
                          'What to do with the clients\' relative weights.')
  flags.DEFINE_float('alpha', 0.1, 'Byzantine proportion.')
  flags.DEFINE_float('alpha_star', 0.5, 'Byzantine weight proportion.')
  flags.DEFINE_multi_enum('metrics', list(available_metrics), list(available_metrics),
                          'How to compare the results of preprocess to the original weights.')
  flags.DEFINE_bool('check_validity', True, 'Whether to check if a given preprocess solution gives a valid mwp.')
  flags.DEFINE_bool('plot', True, 'Whether to plot the different weights.')
  flags.DEFINE_bool('compare_mwp', True, 'Whether to print the mwp of each preprocess output.')

flags.register_validator('limit_count', create_optional_validator(check_positive))
flags.register_validator('alpha', check_proportion)
flags.register_validator('alpha_star', check_proportion)

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
    plot_weights({'original': original_weights, **named_new_weights})
  # comparing mwp
  if compare_mwp:
    print('mwp')
    pp.pprint({preprocess_name: maximal_weight_proportion(new_weights, alpha)
               for preprocess_name, new_weights in named_new_weights.items()})
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
  weights = list(weights)
  weights = np.array(weights)
  # applying different preprocess procedures
  selected_preprocess_constructors = {name: PREPROC_TRANSFORMS[name] for name in FLAGS.preprocess_funcs}
  selected_preprocess = {name: constructor(alpha=FLAGS.alpha, alpha_star=FLAGS.alpha_star)
                         for name, constructor in selected_preprocess_constructors.items()}
  named_new_weights = {name: preprocess.fit_transform(weights)
                       for name, preprocess in selected_preprocess.items()}

  # comparing the outputs
  selected_metrics = {name: available_metrics[name] for name in FLAGS.metrics}
  compare_weights(weights, named_new_weights, selected_metrics,
                  FLAGS.alpha, FLAGS.alpha_star, FLAGS.check_validity, FLAGS.plot, FLAGS.compare_mwp)


if __name__ == '__main__':
  app.run(main)
