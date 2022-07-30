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

import functools
import os

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from absl import app, flags

from tensorflow_federated.python.learning import ClientWeighting
from tensorflow_federated.python.simulation.baselines import ClientSpec

from google_tff_research.utils import task_utils, utils_impl, training_loop
from google_tff_research.utils.optimizers import optimizer_utils
from preprocess import PREPROC_TRANSFORMS
from aggregators import trimmed_mean, median, mean, NumpyAggregationFactory, PreprocessedAggregationFactory
from flags_validators import check_positive, check_non_negative, check_proportion, check_integer, create_or_validator

from attacks.local import ATTACKS
from tff_patch import build_federated_averaging_process, compose_dataset_computation_with_iterative_process

CLIENT_WEIGHTING = ['uniform', 'num_examples']
AGGREGATORS = ['mean', 'median', 'trimmed_mean']
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
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer(
    'rounds_per_eval', 1,
    'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

  # Parameters specific for our paper
  flags.DEFINE_enum('weight_preproc', 'num_examples', CLIENT_WEIGHTING + list(PREPROC_TRANSFORMS),
                    'What to do with the clients\' relative weights.')

  flags.DEFINE_enum('aggregation', 'mean', AGGREGATORS, 'select aggregation type to use')

  flags.DEFINE_enum('attack', 'sign_flip', list(ATTACKS), 'select attack type')
  flags.DEFINE_float('num_byzantine', 0.1, 'select either the proportion or the number of byzantine clients', 0.0)
  flags.DEFINE_enum('byzantines_part_of', 'total', BYZANTINES_PART_OF,
                    'select whether num_clients are takes as part of the total amount of clients or in each round')
  flags.DEFINE_integer('byzantine_client_weight', 1_000_000, 'select fake client weight byzantine client publish')
  flags.DEFINE_float('alpha', 0.1, 'select Byzantine proportion')
  flags.DEFINE_float('alpha_star', 0.5, 'select Byzantine weight proportion')

with utils_impl.record_hparam_flags() as task_flags:
  task_utils.define_task_flags()


flags.register_validator('client_epochs_per_round', check_positive)
flags.register_validator('client_batch_size', check_positive)
flags.register_validator('clients_per_round', check_positive)
flags.register_validator('total_rounds', check_positive)
flags.register_validator('rounds_per_eval', check_positive)
flags.register_validator('rounds_per_checkpoint', check_positive)
flags.register_validator('num_byzantine', check_non_negative)
flags.register_validator('num_byzantine', create_or_validator(check_proportion, check_integer))
flags.register_validator('byzantine_client_weight', check_non_negative)
flags.register_validator('alpha', check_proportion)
flags.register_validator('alpha_star', check_proportion)


flags.mark_flags_as_required(['experiment_name', 'task'])

FLAGS = flags.FLAGS


def _write_hparam_flags():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task-specific flags.
  task_flag_dict = utils_impl.lookup_flag_values(task_flags)
  hparam_dict.update(task_flag_dict)
  results_dir = os.path.join(FLAGS.root_output_dir, 'results',
                             FLAGS.experiment_name)
  utils_impl.create_directory_if_not_exists(results_dir)
  hparam_file = os.path.join(results_dir, 'hparams.csv')
  utils_impl.atomic_write_series_to_csv(hparam_dict, hparam_file)


def configure_task():
  train_client_spec = ClientSpec(num_epochs=FLAGS.client_epochs_per_round, batch_size=FLAGS.client_batch_size)
  task = task_utils.create_task_from_flags(train_client_spec)
  model_fn = task.model_fn
  train_data = task.datasets.train_data.preprocess(task.datasets.train_preprocess_fn)
  test_data = task.datasets.get_centralized_test_data()
  return model_fn, train_data, test_data


def configure_client_weight_fn():
  client_weight_fn = None
  if FLAGS.weight_preproc == 'num_examples' and FLAGS.task in task_utils.TASKS_NUM_TOKENS:
    def client_weight_fn(local_outputs):
      return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)
  elif FLAGS.weight_preproc == 'uniform':
    client_weight_fn = ClientWeighting.UNIFORM
  return client_weight_fn


def configure_inner_aggregator():
  inner_aggregator = mean
  inner_aggregators = {'trimmed_mean': functools.partial(trimmed_mean, beta=FLAGS.alpha),
                       'median': median,
                       'mean': mean}
  if FLAGS.aggregation in inner_aggregators:
    inner_aggregator = inner_aggregators[FLAGS.aggregation]
  return inner_aggregator


def configure_aggregator(train_data):
  inner_aggregator = configure_inner_aggregator()
  if FLAGS.weight_preproc in PREPROC_TRANSFORMS:
    preproc_constructor = PREPROC_TRANSFORMS[FLAGS.weight_preproc]
    preproc_transform = preproc_constructor(alpha=FLAGS.alpha, alpha_star=FLAGS.alpha_star)
    if FLAGS.byzantines_part_of == 'total':
      preproc_transform.fit(train_data.datasets())

    preprocesses = {'total': lambda weights: preproc_transform.transform(weights),
                    'round': lambda weights: preproc_transform.fit_transform(weights)}

    PreprocessedAggregationFactory(NumpyAggregationFactory(inner_aggregator),
                                   preprocesses[FLAGS.byzantines_part_of])
  else:
    if FLAGS.aggregation == 'mean':
      aggregator = None  # defaults to reduce mean
    else:
      aggregator = NumpyAggregationFactory(inner_aggregator)
  return aggregator


def configure_attack():
  attack_constructor = ATTACKS[FLAGS.attack]
  attack = attack_constructor()
  return attack


def configure_iterative_process(model_fn, train_data):
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
  client_weight_fn = configure_client_weight_fn()
  aggregator = configure_aggregator(train_data)
  attack = configure_attack()
  iterative_process = build_federated_averaging_process(model_fn=model_fn,
                                                        client_optimizer_fn=client_optimizer_fn,
                                                        server_optimizer_fn=server_optimizer_fn,
                                                        client_weighting=client_weight_fn,
                                                        model_update_aggregation_factory=aggregator,
                                                        byzantine_client_weight=FLAGS.byzantine_client_weight,
                                                        attack=attack)

  @tff.tf_computation((tf.string, tf.bool))
  def build_train_dataset_from_client_id(client_id_with_byzflag):
    client_id, byzflag = client_id_with_byzflag
    client_dataset = train_data.dataset_computation(client_id)
    return client_dataset, byzflag

  training_process = compose_dataset_computation_with_iterative_process(
    build_train_dataset_from_client_id, iterative_process)
  training_process.get_model_weights = iterative_process.get_model_weights
  return training_process


def configure_num_byzantine(client_ids):
  max_sizes = {'total': len(client_ids), 'round': FLAGS.clients_per_round}
  num_byzantine = FLAGS.num_byzantine
  max_size = max_sizes[FLAGS.byzantines_part_of]
  if num_byzantine < 1:
    num_byzantine = num_byzantine * max_size
  num_byzantine = int(num_byzantine)
  if num_byzantine > max_size:
    raise ValueError(f'num_byzantine is larger than the number of clients. '
                     f'num_byzantine = {num_byzantine}, '
                     f'number of clients = {max_size}')
  return num_byzantine


def configure_client_datasets_fn(train_data):
  client_ids = train_data.client_ids
  client_ids_fn = tff.simulation.build_uniform_sampling_fn(
    client_ids, replace=False, random_seed=FLAGS.client_datasets_random_seed)

  num_byzantine = configure_num_byzantine(client_ids)
  chosen_byz_ids = set(np.random.choice(client_ids, num_byzantine, False))

  def client_sampling_fn_with_byzantine(round_num):
    chosen_client_ids = list(client_ids_fn(round_num, FLAGS.clients_per_round))
    byz_mask = np.zeros(FLAGS.clients_per_round, dtype=np.bool)
    if FLAGS.byzantines_part_of == 'round':
      byzantine_indices = np.random.choice(np.arange(FLAGS.clients_per_round), num_byzantine, False)
      byz_mask[byzantine_indices] = True
    elif FLAGS.byzantines_part_of == 'total':
      byz_mask = np.array([client_id in chosen_byz_ids for client_id in chosen_client_ids], dtype=np.bool)

    return list(zip(chosen_client_ids, byz_mask))
  return client_sampling_fn_with_byzantine


def configure_evaluation_fns(model_fn, test_data, get_model_weights):
  evaluate_fn = tff.learning.build_federated_evaluation(model_fn)

  def test_fn(state):
    return evaluate_fn(get_model_weights(state), [test_data])

  def validation_fn(state, round_num):
    del round_num
    return evaluate_fn(get_model_weights(state), [test_data])

  return validation_fn, test_fn


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  model_fn, train_data, test_data = configure_task()

  training_process = configure_iterative_process(model_fn, train_data)

  client_datasets_fn = configure_client_datasets_fn(train_data)

  validation_fn, test_fn = configure_evaluation_fns(model_fn, test_data, training_process.get_model_weights)

  _write_hparam_flags()

  training_loop.run(
    iterative_process=training_process,
    client_datasets_fn=client_datasets_fn,
    validation_fn=validation_fn,
    test_fn=test_fn,
    total_rounds=FLAGS.total_rounds,
    experiment_name=FLAGS.experiment_name,
    root_output_dir=FLAGS.root_output_dir,
    rounds_per_eval=FLAGS.rounds_per_eval,
    rounds_per_checkpoint=FLAGS.rounds_per_checkpoint)


if __name__ == '__main__':
  app.run(main)
