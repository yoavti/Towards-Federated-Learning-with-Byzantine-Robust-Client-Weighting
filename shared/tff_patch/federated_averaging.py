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

from typing import Callable, Optional, Any, Union

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model as model_lib

from shared.tff_patch import optimizer_utils
from shared.tff_patch.byzantine_weight_client_fed_avg import ByzantineWeightClientFedAvg


def _default_server_optimizer_fn():
  return tf.keras.optimizers.SGD(learning_rate=1.0)


def build_federated_averaging_process(
    model_fn: Callable[[], model_lib.Model],
    client_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
    server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer] = _default_server_optimizer_fn,
    *,  # Require named (non-positional) parameters for the following kwargs:
    client_weighting: Optional[Union[client_weight_lib.ClientWeightType, Callable[[Any], tf.Tensor]]] = None,
    model_update_aggregation_factory: Optional[factory.WeightedAggregationFactory] = None,
    byzantine_client_weight: int = 1_000_000
) -> iterative_process.IterativeProcess:
  """Builds an iterative process that performs federated averaging.

  This function creates a `tff.templates.IterativeProcess` that performs
  federated averaging on client models. The iterative process has the following
  methods inherited from `tff.templates.IterativeProcess`:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a `tff.learning.framework.ServerState`
      representing the initial state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>)` where `S` is a
      `tff.learning.framework.ServerState` whose type matches that of the output
      of `initialize`, and `{B*}@CLIENTS` represents the client datasets, where
      `B` is the type of a single batch. This computation returns a
      `tff.learning.framework.ServerState` representing the updated server state
      and metrics that are the result of
      `tff.learning.Model.federated_output_computation` during client training
      and any other metrics from broadcast and aggregation processes.

  The iterative process also has the following method not inherited from
  `tff.templates.IterativeProcess`:

  *   `get_model_weights`: A `tff.Computation` that takes as input the
      a `tff.learning.framework.ServerState`, and returns a
      `tff.learning.ModelWeights` containing the state's model weights.

  Each time the `next` method is called, the server model is broadcast to each
  client using a broadcast function. For each client, one epoch of local
  training is performed via the `tf.keras.optimizers.Optimizer.apply_gradients`
  method of the client optimizer. Each client computes the difference between
  the client model after training and the initial broadcast model. These model
  deltas are then aggregated at the server using some aggregation function. The
  aggregate model delta is applied at the server by using the
  `tf.keras.optimizers.Optimizer.apply_gradients` method of the server
  optimizer.

  Note: the default server optimizer function is `tf.keras.optimizers.SGD`
  with a learning rate of 1.0, which corresponds to adding the model delta to
  the current server model. This recovers the original FedAvg algorithm in
  [McMahan et al., 2017](https://arxiv.org/abs/1602.05629). More
  sophisticated federated averaging procedures may use different learning rates
  or server optimizers.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    client_optimizer_fn: A no-arg callable that returns a `tf.keras.Optimizer`.
    server_optimizer_fn: A no-arg callable that returns a `tf.keras.Optimizer`.
      By default, this uses `tf.keras.optimizers.SGD` with a learning rate of
      1.0.
    client_weighting: A value of `tff.learning.ClientWeighting` that specifies a
      built-in weighting method, or a callable that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If None, defaults to weighting
      by number of examples.
    model_update_aggregation_factory: An optional
      `tff.aggregators.WeightedAggregationFactory` or
      `tff.aggregators.UnweightedAggregationFactory` that constructs
      `tff.templates.AggregationProcess` for aggregating the client model
      updates on the server. If `None`, uses `tff.aggregators.MeanFactory`. Must
      be `None` if `aggregation_process` is not `None.`
    byzantine_client_weight: Number of samples each Byzantine client reports.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  model_to_client_delta_fn = ByzantineWeightClientFedAvg.model_to_client_delta_fn(
    client_optimizer_fn,
    client_weighting=client_weighting,
    byzantine_client_weight=byzantine_client_weight)

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
