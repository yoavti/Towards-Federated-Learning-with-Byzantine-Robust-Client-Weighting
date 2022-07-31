# Copyright 2018, The TensorFlow Federated Authors.
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
"""An implementation of the Federated Averaging algorithm.

Based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

from typing import Callable, Optional

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.learning import model as model_lib

from shared.tff_patch import optimizer_utils


def _default_server_optimizer_fn():
  return tf.keras.optimizers.SGD(learning_rate=1.0)


def build_federated_averaging_process(
    model_fn: Callable[[], model_lib.Model],
    model_to_client_delta_fn: Callable[[Callable[[], model_lib.Model]], optimizer_utils.ClientDeltaFn],
    server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer] = _default_server_optimizer_fn,
    model_update_aggregation_factory: Optional[factory.WeightedAggregationFactory] = None
) -> iterative_process.IterativeProcess:
  """Builds an iterative process that performs federated averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    model_to_client_delta_fn: A function from a `model_fn` to a `ClientDeltaFn`.
    server_optimizer_fn: A no-arg callable that returns a `tf.keras.Optimizer`.
      By default, this uses `tf.keras.optimizers.SGD` with a learning rate of
      1.0.
    model_update_aggregation_factory: An optional
      `tff.aggregators.WeightedAggregationFactory` or
      `tff.aggregators.UnweightedAggregationFactory` that constructs
      `tff.templates.AggregationProcess` for aggregating the client model
      updates on the server. If `None`, uses `tff.aggregators.MeanFactory`. Must
      be `None` if `aggregation_process` is not `None.`

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  iter_proc = optimizer_utils.build_model_delta_optimizer_process(
    model_fn,
    model_to_client_delta_fn=model_to_client_delta_fn,
    server_optimizer_fn=server_optimizer_fn,
    model_update_aggregation_factory=model_update_aggregation_factory)

  server_state_type = iter_proc.state_type.member

  @computations.tf_computation(server_state_type)
  def get_model_weights(server_state):
    return server_state.model

  iter_proc.get_model_weights = get_model_weights
  return iter_proc
