import tensorflow as tf
import tensorflow_federated as tff

from shared.aggregators.utils import ds_to_array, init_fn


class PreprocessedAggregationFactory(tff.aggregators.WeightedAggregationFactory):
  def __init__(self, aggregation_factory, preprocess):
    self._aggregation_factory = aggregation_factory
    self._preprocess = preprocess

  def create(self, value_type, weight_type):
    _aggregation_process = self._aggregation_factory.create()
    _initialize_fn = _aggregation_process.initialize
    _next_fn = _aggregation_process.next

    @tff.tf_computation
    def numpy_bridge(weights):
      w = ds_to_array(weights)
      return tf.reshape(tf.numpy_function(self._preprocess, [w], tf.float32), tf.shape(w)[1:])

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type),
                               tff.type_at_clients(weight_type))
    def next_fn(state, value, weight):
      return _next_fn(state, value, numpy_bridge(weight))

    return tff.templates.AggregationProcess(_initialize_fn, next_fn)
