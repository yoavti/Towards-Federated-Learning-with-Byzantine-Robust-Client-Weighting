import tensorflow as tf
import tensorflow_federated as tff


class NumpyAggregationFactory(tff.aggregators.WeightedAggregationFactory):
  def __init__(self, numpy_fn):
    self._numpy_fn = numpy_fn

  def create(self, value_type, weight_type):

    @tff.tf_computation
    def numpy_bridge(values, weights):
      v = ds_to_array(values)
      w = ds_to_array(weights)
      return tf.reshape(tf.numpy_function(self._numpy_fn, [v, w], tf.float32), tf.shape(v)[1:])

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type),
                               tff.type_at_clients(weight_type))
    def next_fn(state, value, weight):
      weights = tff.federated_collect(weight)

      weighted_mean_value = tff.federated_zip([
        tff.federated_map(numpy_bridge,
                          (tff.federated_collect(tff.federated_map(tff.tf_computation(lambda _: _[idx]), value)),
                           weights))
        for idx in range(len(value_type))
      ])
      no_metrics = tff.federated_value((), tff.SERVER)
      return tff.templates.MeasuredProcessOutput(state, weighted_mean_value, no_metrics)

    return tff.templates.AggregationProcess(init_fn, next_fn)


@tff.federated_computation()
def init_fn():
  return tff.federated_value((), tff.SERVER)


@tf.function
def ds_to_array(dataset):
  # annoyingly tf graph mode requires the following
  # in order to extract a tensor from the dataset
  # + it requires int64 converting to int32 in the TensorArray API
  tensor_array = tf.TensorArray(dtype=dataset.element_spec.dtype,
                                size=tf.cast(tf.data.experimental.cardinality(dataset), tf.int32),
                                element_shape=dataset.element_spec.shape)
  for i, t in dataset.enumerate():
    tensor_array = tensor_array.write(tf.cast(i, tf.int32), t)
  return tensor_array.stack()
