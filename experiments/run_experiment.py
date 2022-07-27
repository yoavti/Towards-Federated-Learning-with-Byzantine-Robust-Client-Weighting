# This file was adapted from https://github.com/google-research/federated:
#
# Copyright 2020, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import functools
import os.path
from typing import Callable

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.learning import ClientWeighting

from shared.aggregators import trimmed_mean, median, mean
from shared.truncate import truncate
from shared.lp import lp
from google_tff_research.optimization.shared import optimizer_utils, training_specs
from experiments import federated_shakespeare, federated_stackoverflow
import tff_patch as tff_patch
from experiments.numpy_aggr import NumpyAggrFactory
from experiments.attacks.local import ConstantAttack, GaussianAttack, NoAttack, RandomSignFlipAttack, SignFlipAttack
from google_tff_research.utils import training_loop, utils_impl

SUPPORTED_TASKS = ['shakespeare', 'stackoverflow_nwp']
CLIENT_WEIGHTING = {'uniform': ClientWeighting.UNIFORM, 'num_examples': ClientWeighting.NUM_EXAMPLES}
PREPROC_FUNCS = {'truncate': truncate, 'lp': lp}
AGGREGATORS = ['mean', 'median', 'trimmed_mean']
ATTACKS = {'none': NoAttack, 'sign_flip': SignFlipAttack, 'constant': ConstantAttack, 'gaussian': GaussianAttack,
           'random_sign_flip': RandomSignFlipAttack}  # delta_to_zero
BYZANTINES_PART_OF = ['total', 'round']

with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')

  # Training loop configuration
  flags.DEFINE_string(
    'experiment_name', None, 'The name of this experiment. Will be append to '
                             '--root_output_dir to separate experiment results.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer(
    'rounds_per_eval', 1,
    'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

  # Parameters specific for our paper
  flags.DEFINE_enum('weight_preproc', 'num_examples', list(CLIENT_WEIGHTING) + list(PREPROC_FUNCS),
                    'What to do with the clients\' relative weights.')

  flags.DEFINE_enum('aggregation', 'mean', AGGREGATORS, 'select aggregation type to use')

  flags.DEFINE_enum('attack', 'none', list(ATTACKS), 'select attack type')
  flags.DEFINE_float('num_byzantine', 0.1, 'select either the proportion or the number of byzantine clients', 0.0)
  flags.DEFINE_enum('byzantines_part_of', 'total', BYZANTINES_PART_OF,
                    'select whether num_clients are takes as part of the total amount of clients or in each round')
  flags.DEFINE_integer('byzantine_client_weight', 1_000_000, 'select fake client weight byzantine client publish')
  flags.DEFINE_float('alpha', 0.1, 'select Byzantine proportion')
  flags.DEFINE_float('alpha_star', 0.5, 'select Byzantine weight proportion')

with utils_impl.record_hparam_flags() as task_flags:
  # Task specification
  flags.DEFINE_enum('task', None, SUPPORTED_TASKS,
                    'Which task to perform federated training on.')

with utils_impl.record_hparam_flags() as shakespeare_flags:
  # Shakespeare flags
  flags.DEFINE_integer(
    'shakespeare_sequence_length', 80,
    'Length of character sequences to use for the RNN model.')

with utils_impl.record_hparam_flags() as so_nwp_flags:
  # Stack Overflow NWP flags
  flags.DEFINE_integer('so_nwp_vocab_size', 10000, 'Size of vocab to use.')
  flags.DEFINE_integer('so_nwp_num_oov_buckets', 1,
                       'Number of out of vocabulary buckets.')
  flags.DEFINE_integer('so_nwp_sequence_length', 20,
                       'Max sequence length to use.')
  flags.DEFINE_integer('so_nwp_max_elements_per_user', 1000, 'Max number of '
                       'training sentences to use per user.')
  flags.DEFINE_integer(
      'so_nwp_num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')

FLAGS = flags.FLAGS

TASK_FLAGS = collections.OrderedDict(
  shakespeare=shakespeare_flags,
  stackoverflow_nwp=so_nwp_flags,
)


def _write_hparam_flags():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task-specific flags.
  task_name = FLAGS.task
  if task_name in TASK_FLAGS:
    task_hparam_dict = utils_impl.lookup_flag_values(TASK_FLAGS[task_name])
    hparam_dict.update(task_hparam_dict)

  results_dir = os.path.join(FLAGS.root_output_dir, 'results',
                             FLAGS.experiment_name)
  utils_impl.create_directory_if_not_exists(results_dir)
  hparam_file = os.path.join(results_dir, 'hparams.csv')
  utils_impl.atomic_write_series_to_csv(hparam_dict, hparam_file)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  # If GPU is provided, TFF will by default use the first GPU like TF. The
  # following lines will configure TFF to use multi-GPUs and distribute client
  # computation on the GPUs. Note that we put server computatoin on CPU to avoid
  # potential out of memory issue when a large number of clients is sampled per
  # round. The client devices below can be an empty list when no GPU could be
  # detected by TF.
  # client_devices = tf.config.list_logical_devices('GPU')
  # server_device = tf.config.list_logical_devices('CPU')[0]
  # tff.backends.native.set_local_execution_context(
  #     server_tf_device=server_device, client_tf_devices=client_devices)

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  def iterative_process_builder(
          model_fn: Callable[[],
                             tff.learning.Model]) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`.

    Args:
      model_fn: A no-arg function returning a `tff.learning.Model`.

    Returns:
      A `tff.templates.IterativeProcess`.
    """
    client_weight_fn = None
    if FLAGS.task in SUPPORTED_TASKS and FLAGS.weight_preproc == 'num_examples':

      def client_weight_fn(local_outputs):
        return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)
    elif FLAGS.weight_preproc in CLIENT_WEIGHTING:
      client_weight_fn = CLIENT_WEIGHTING[FLAGS.weight_preproc]

    if FLAGS.aggregation == 'trimmed_mean':
      inner_aggregator = functools.partial(trimmed_mean, beta=FLAGS.alpha)
    elif FLAGS.aggregation == 'median':
      inner_aggregator = median
    elif FLAGS.aggregation == 'mean':
      inner_aggregator = mean
    else:
      inner_aggregator = mean

    if FLAGS.weight_preproc in PREPROC_FUNCS:

      def aggregate_with_preproc(points, weights):
        return inner_aggregator(points,
                                PREPROC_FUNCS[FLAGS.weight_preproc](weights,
                                                                    alpha=FLAGS.alpha,
                                                                    alpha_star=FLAGS.alpha_star))

      aggregator = NumpyAggrFactory(aggregate_with_preproc)
    else:
      if FLAGS.aggregation == 'mean':
        aggregator = None  # defaults to reduce mean
      else:
        aggregator = NumpyAggrFactory(inner_aggregator)

    attack = ATTACKS[FLAGS.attack]()

    return tff_patch.build_federated_averaging_process(
      model_fn=model_fn,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weighting=client_weight_fn,
      model_update_aggregation_factory=aggregator,
      byzantine_client_weight=FLAGS.byzantine_client_weight,
      attack=attack,
    )

  task_spec = training_specs.TaskSpec(
    iterative_process_builder=iterative_process_builder,
    client_epochs_per_round=FLAGS.client_epochs_per_round,
    client_batch_size=FLAGS.client_batch_size,
    clients_per_round=FLAGS.clients_per_round,
    client_datasets_random_seed=FLAGS.client_datasets_random_seed)

  if FLAGS.num_byzantine >= 1. and not FLAGS.num_byzantine.is_integer():
    raise ValueError('num_byzantine must either be a proportion (i.e. [0, 1)) or a full number')

  if FLAGS.task == 'shakespeare':
    runner_spec = federated_shakespeare.configure_training(
      task_spec,
      sequence_length=FLAGS.shakespeare_sequence_length,
      num_byzantine=FLAGS.num_byzantine,
      byzantines_part_of=FLAGS.byzantines_part_of)
  elif FLAGS.task == 'stackoverflow_nwp':
    runner_spec = federated_stackoverflow.configure_training(
      task_spec,
      vocab_size=FLAGS.so_nwp_vocab_size,
      num_oov_buckets=FLAGS.so_nwp_num_oov_buckets,
      sequence_length=FLAGS.so_nwp_sequence_length,
      max_elements_per_user=FLAGS.so_nwp_max_elements_per_user,
      num_validation_examples=FLAGS.so_nwp_num_validation_examples,
      num_byzantine=FLAGS.num_byzantine,
      byzantines_part_of=FLAGS.byzantines_part_of)
  else:
    raise ValueError(
      '--task flag {} is not supported, must be one of {}.'.format(
        FLAGS.task, SUPPORTED_TASKS))

  _write_hparam_flags()

  training_loop.run(
    iterative_process=runner_spec.iterative_process,
    client_datasets_fn=runner_spec.client_datasets_fn,
    validation_fn=runner_spec.validation_fn,
    test_fn=runner_spec.test_fn,
    total_rounds=FLAGS.total_rounds,
    experiment_name=FLAGS.experiment_name,
    root_output_dir=FLAGS.root_output_dir,
    rounds_per_eval=FLAGS.rounds_per_eval,
    rounds_per_checkpoint=FLAGS.rounds_per_checkpoint)


if __name__ == '__main__':
  app.run(main)
