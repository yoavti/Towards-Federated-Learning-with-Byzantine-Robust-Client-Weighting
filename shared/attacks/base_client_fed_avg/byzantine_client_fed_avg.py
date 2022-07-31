import collections

from typing import Callable, Any, Union

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.tensorflow_libs import tensor_utils

from shared.tff_patch.federated_averaging import ClientOutput, ClientDeltaFn


class ByzantineClientFedAvg(ClientDeltaFn):
  """Client TensorFlow logic for Federated Averaging with Byzantine clients."""

  def __init__(
      self,
      model: model_lib.Model,
      optimizer: tf.keras.optimizers.Optimizer,
      client_weighting: Union[client_weight_lib.ClientWeightType,
                              Callable[[Any], tf.Tensor]] = client_weight_lib.ClientWeighting.NUM_EXAMPLES):
    """Creates the client computation for Federated Averaging.

    Note: All variable creation required for the client computation (e.g. model
    variable creation) must occur in during construction, and not during
    `__call__`.

    Args:
      model: A `tff.learning.Model` instance.
      optimizer: A `tf.keras.Optimizer` instance.
      client_weighting: A value of `tff.learning.ClientWeighting` that
        specifies a built-in weighting method, or a callable that takes the
        output of `model.report_local_outputs` and returns a tensor that
        provides the weight in the federated average of model deltas.
    """
    py_typecheck.check_type(model, model_lib.Model)
    self._model = model_utils.enhance(model)
    self._optimizer = optimizer
    py_typecheck.check_type(self._model, model_utils.EnhancedModel)
    if not isinstance(client_weighting, client_weight_lib.ClientWeighting) and not callable(client_weighting):
      raise TypeError('`client_weighting` must be either instance of `ClientWeighting` or callable. '
                      f'Found type {type(client_weighting)}.')
    self._client_weighting = client_weighting
    self._dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(False)

  @property
  def variables(self):
    return []

  @tf.function
  def __call__(self, dataset_with_byzflag, initial_weights):
    model = self._model
    optimizer = self._optimizer
    tf.nest.map_structure(lambda a, b: a.assign(b), model.weights, initial_weights)

    def reduce_fn(num_examples_sum, batch):
      """Train `tff.learning.Model` on local client batch."""
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)

      gradients = tape.gradient(output.loss, model.weights.trainable)
      optimizer.apply_gradients(zip(gradients, model.weights.trainable))

      if output.num_examples is None:
        return num_examples_sum + tf.shape(
          output.predictions, out_type=tf.int64)[0]
      else:
        return num_examples_sum + tf.cast(output.num_examples, tf.int64)

    dataset, byzflag = dataset_with_byzflag

    num_examples_sum = self._dataset_reduce_fn(reduce_fn, dataset,
                                               initial_state_fn=lambda: tf.zeros(shape=[], dtype=tf.int64))

    weights_delta = tf.nest.map_structure(tf.subtract, model.weights.trainable, initial_weights.trainable)
    model_output = model.report_local_outputs()

    weights_delta, has_non_finite_delta = tensor_utils.zero_all_if_any_non_finite(weights_delta)
    # Zero out the weight if there are any non-finite values.
    if has_non_finite_delta > 0:
      weights_delta_weight = tf.constant(0.0)
    elif self._client_weighting is client_weight_lib.ClientWeighting.NUM_EXAMPLES:
      weights_delta_weight = tf.cast(num_examples_sum, tf.float32)
    elif self._client_weighting is client_weight_lib.ClientWeighting.UNIFORM:
      weights_delta_weight = tf.constant(1.0)
    else:
      weights_delta_weight = self._client_weighting(model_output)
    optimizer_output = collections.OrderedDict(num_examples=num_examples_sum)
    return ClientOutput(weights_delta, weights_delta_weight, model_output, optimizer_output)

  @staticmethod
  def model_to_client_delta_fn(
      client_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
      *,
      client_weighting: Union[client_weight_lib.ClientWeightType,
                              Callable[[Any], tf.Tensor]] = client_weight_lib.ClientWeighting.NUM_EXAMPLES
  ) -> Callable[[Callable[[], model_lib.Model]], ClientDeltaFn]:
    """Returns a function that accepts a model creation function and returns a `ClientDeltaFn` instance.

        Args:
          client_optimizer_fn: A no-arg callable that returns a `tf.keras.Optimizer`.
          client_weighting: A value of `tff.learning.ClientWeighting` that specifies a
            built-in weighting method, or a callable that takes the output of
            `model.report_local_outputs` and returns a tensor that provides the weight
            in the federated average of model deltas. If None, defaults to weighting
            by number of examples.

        Returns:
          A function that accepts a model creation function and returns a `ClientDeltaFn` instance."""

    def ret(model_fn: Callable[[], model_lib.Model]) -> ClientDeltaFn:
      return ByzantineClientFedAvg(model_fn(), client_optimizer_fn(), client_weighting)

    return ret
