# Copyright 2019, Krishna Pillutla and Sham M. Kakade and Zaid Harchaoui.
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
"""Simple implementation of the RFA Algorithm for robust aggregation."""

import tensorflow as tf
import tensorflow_federated as tff

from shared.aggregators.utils import init_fn


class RobustWeiszfeldFactory(tff.aggregators.WeightedAggregationFactory):
  """Aggregator for the Robust Federated Aggregation algorithm."""

  def __init__(self, num_communication_passes=5, tolerance=1e-6):
    """Initializes RobustWeiszfeldFactory.

    Args:
      num_communication_passes: Number of communication rounds in the smoothed
        Weiszfeld algorithm (min. 1).
      tolerance: Smoothing parameter of smoothed Weiszfeld algorithm. Default
        1e-6.
    """
    if not isinstance(num_communication_passes, int):
      raise TypeError('Expected `int`, found {}.'.format(
          type(num_communication_passes)))
    if num_communication_passes < 1:
      raise ValueError('Aggregation requires num_communication_passes >= 1')

    self._num_communication_passes = num_communication_passes
    self._tolerance = tolerance

  def create(self, value_type, weight_type):

    @tff.tf_computation(tf.float32, value_type, value_type)
    def update_weight_fn(weight, server_model, client_model):
      sqnorms = tf.nest.map_structure(lambda a, b: tf.norm(a - b)**2,
                                      server_model, client_model)
      sqnorm = tf.reduce_sum(sqnorms)
      return tf.math.divide_no_nan(
          weight, tf.math.maximum(self._tolerance, tf.math.sqrt(sqnorm)))

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type),
                               tff.type_at_clients(weight_type))
    def next_fn(state, value, weight):
      aggregate = tff.federated_mean(value, weight=weight)
      for _ in range(self._num_communication_passes - 1):
        aggregate_at_client = tff.federated_broadcast(aggregate)
        updated_weight = tff.federated_map(update_weight_fn,
                                           (weight, aggregate_at_client, value))
        aggregate = tff.federated_mean(value, weight=updated_weight)
      no_metrics = tff.federated_value((), tff.SERVER)
      return tff.templates.MeasuredProcessOutput(state, aggregate, no_metrics)

    return tff.templates.AggregationProcess(init_fn, next_fn)
