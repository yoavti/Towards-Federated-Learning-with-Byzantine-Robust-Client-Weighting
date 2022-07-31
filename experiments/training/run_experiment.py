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

import os

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from absl import app, flags

from shared.google_tff_research.utils import task_utils, utils_impl, training_loop
from shared.google_tff_research.utils.optimizers import optimizer_utils

from shared.flags_validators import check_positive, check_non_negative, check_proportion, check_integer, create_or_validator
from shared.extract_client_weights import extract_weights
from shared.tff_patch.federated_averaging import build_model_delta_optimizer_process
from shared.tff_patch.iterative_process_compositions import compose_dataset_computation_with_iterative_process

from shared.aggregators.spec import AggregatorSpec
from shared.attacks.spec import AttackSpec
from shared.preprocess.spec import PreprocessSpec
from shared.client_weighting.spec import ClientWeightingSpec
from tensorflow_federated.python.simulation.baselines import ClientSpec

from shared.aggregators.configure import configure_aggregator
from shared.attacks.configure import configure_attack
from shared.preprocess.configure import configure_preprocess
from shared.client_weighting.configure import configure_client_weight

from shared.attacks.local import LOCAL_ATTACKS
from shared.aggregators import ALL_AGGREGATORS
from shared.client_weighting import CLIENT_WEIGHTING
from shared.preprocess import PREPROC_TRANSFORMS
from shared.byzantines_part_of import BYZANTINES_PART_OF

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
  flags.DEFINE_enum('client_weighting', 'num_examples', CLIENT_WEIGHTING,
                    'What to do with the clients\' relative weights.')
  flags.DEFINE_enum('weight_preproc', 'none', list(PREPROC_TRANSFORMS) + ['none'],
                    'Whether to use the clients\' weight or not.')
  flags.DEFINE_enum('aggregation', 'mean', ALL_AGGREGATORS, 'select aggregation type to use')
  flags.DEFINE_enum('attack', 'sign_flip', list(LOCAL_ATTACKS), 'select attack type')
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


def configure_iterative_process(model_fn, train_data):
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  client_datasets = train_data.datasets()
  weights = extract_weights(client_datasets)

  preprocess_spec = PreprocessSpec(FLAGS.weight_preproc, FLAGS.byzantines_part_of, FLAGS.alpha, FLAGS.alpha_star,
                                   weights)
  preprocess = configure_preprocess(preprocess_spec)

  aggregator_spec = AggregatorSpec(FLAGS.aggregation, preprocess, FLAGS.alpha)
  aggregation_factory = configure_aggregator(aggregator_spec)

  client_weighting_spec = ClientWeightingSpec(FLAGS.client_weighting, FLAGS.task, aggregation_factory)
  client_weight_fn = configure_client_weight(client_weighting_spec)

  attack_spec = AttackSpec(client_optimizer_fn, client_weight_fn, FLAGS.byzantine_client_weight, FLAGS.attack)
  attack = configure_attack(attack_spec)

  iterative_process = build_model_delta_optimizer_process(
    model_fn,
    model_to_client_delta_fn=attack,
    server_optimizer_fn=server_optimizer_fn,
    model_update_aggregation_factory=aggregation_factory)

  server_state_type = iterative_process.state_type.member

  @tff.tf_computation(server_state_type)
  def get_model_weights(server_state):
    return server_state.model

  iterative_process.get_model_weights = get_model_weights

  @tff.tf_computation((tf.string, tf.bool))
  def build_train_dataset_from_client_id(client_id_with_byzflag):
    client_id, byzflag = client_id_with_byzflag
    client_dataset = train_data.dataset_computation(client_id)
    return client_dataset, byzflag

  training_process = compose_dataset_computation_with_iterative_process(
    build_train_dataset_from_client_id, iterative_process)
  training_process.get_model_weights = get_model_weights
  return training_process


def configure_num_byzantine(num_clients):
  max_sizes = {'total': num_clients, 'round': FLAGS.clients_per_round}
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

  num_byzantine = configure_num_byzantine(len(client_ids))
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
