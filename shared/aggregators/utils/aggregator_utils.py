import tensorflow as tf
import tensorflow_federated as tff


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
