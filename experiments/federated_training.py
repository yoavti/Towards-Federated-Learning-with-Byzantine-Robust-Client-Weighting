# Copyright 2019, Google LLC.
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
"""Federated Shakespeare next character prediction library using TFF."""

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.simulation.baselines import BaselineTask

from tff_patch import compose_dataset_computation_with_iterative_process
from google_tff_research.optimization.shared import training_specs


def configure_training(task_spec: training_specs.TaskSpec, task: BaselineTask) -> training_specs.RunnerSpec:
  """Configures training for the Shakespeare next-character prediction task.

  This method will load and pre-process datasets and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process compatible with `federated_research.utils.training_loop`.

  Args:
    task_spec: A `TaskSpec` class for creating federated training tasks.
    task: A `BaselineTask` class providing model_fn and datasets.

  Returns:
    A `RunnerSpec` containing attributes used for running the newly created
    federated task.
  """

  train_data = task.datasets.train_data.preprocess(task.datasets.train_preprocess_fn)
  test_data = task.datasets.get_centralized_test_data()

  model_fn = task.model_fn

  iterative_process = task_spec.iterative_process_builder(model_fn)

  @tff.tf_computation((tf.string, tf.bool))
  def build_train_dataset_from_client_id(client_id_with_byzflag):
    client_id, byzflag = client_id_with_byzflag
    client_dataset = train_data.dataset_computation(client_id)
    return client_dataset, byzflag

  training_process = compose_dataset_computation_with_iterative_process(
      build_train_dataset_from_client_id, iterative_process)
  client_ids = train_data.client_ids
  client_ids_fn = tff.simulation.build_uniform_sampling_fn(client_ids, replace=False,
                                                           random_seed=task_spec.client_datasets_random_seed)

  num_byzantine = task_spec.num_byzantine

  if num_byzantine < 1:
    num_byzantine = num_byzantine * len(client_ids)

  num_byzantine = int(num_byzantine)

  if num_byzantine >= len(client_ids):
    raise ValueError(f'num_byzantine is larger than the number of all clients. '
                     f'num_byzantine = {num_byzantine}, '
                     f'number of clients = {len(client_ids)}')

  chosen_byz_ids = set(np.random.choice(client_ids, num_byzantine, False))

  def client_sampling_fn_with_byzantine(round_num):
    chosen_client_ids = list(client_ids_fn(round_num, task_spec.clients_per_round))
    byz_mask = np.zeros(task_spec.clients_per_round, dtype=np.bool)
    if task_spec.byzantines_part_of == 'round':
      byzantine_indices = np.random.choice(np.arange(task_spec.clients_per_round), num_byzantine, False)
      byz_mask[byzantine_indices] = True
    elif task_spec.byzantines_part_of == 'total':
      byz_mask = np.array([client_id in chosen_byz_ids for client_id in chosen_client_ids], dtype=np.bool)

    return list(zip(chosen_client_ids, byz_mask))

  client_sampling_fn = client_sampling_fn_with_byzantine

  training_process.get_model_weights = iterative_process.get_model_weights

  evaluate_fn = tff.learning.build_federated_evaluation(model_fn)

  def test_fn(state):
    return evaluate_fn(
        iterative_process.get_model_weights(state), [test_data])

  def validation_fn(state, round_num):
    del round_num
    return evaluate_fn(
        iterative_process.get_model_weights(state), [test_data])

  return training_specs.RunnerSpec(
      iterative_process=training_process,
      client_datasets_fn=client_sampling_fn,
      validation_fn=validation_fn,
      test_fn=test_fn)
