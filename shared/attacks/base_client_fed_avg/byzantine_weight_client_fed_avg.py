from typing import Callable

import tensorflow as tf

from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model as model_lib

from shared.tff_patch import optimizer_utils
from shared.attacks.base_client_fed_avg.byzantine_client_fed_avg import ByzantineClientFedAvg


class ByzantineWeightClientFedAvg(ByzantineClientFedAvg):
  """Client TensorFlow logic for Federated Averaging with Byzantine clients that report their weight to the server."""

  def __init__(self, model, optimizer,
               client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
               use_experimental_simulation_loop=False, byzantine_client_weight: int = 1_000_000):
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
          use_experimental_simulation_loop: Controls the reduce loop function for
            input dataset. An experimental reduce loop is used for simulation.
          byzantine_client_weight: Number of samples each Byzantine client reports.
        """
    super().__init__(model, optimizer, client_weighting, use_experimental_simulation_loop)
    self._byzantine_client_weight = byzantine_client_weight

  @tf.function
  def __call__(self, dataset_with_byzflag, initial_weights):
    client_output = super().__call__(dataset_with_byzflag, initial_weights)
    weights_delta = client_output.weights_delta
    weights_delta_weight = client_output.weights_delta_weight
    model_output = client_output.model_output
    optimizer_output = client_output.optimizer_output
    dataset, byzflag = dataset_with_byzflag

    if byzflag:
      if self._client_weighting is client_weight_lib.ClientWeighting.NUM_EXAMPLES:
        weights_delta_weight = tf.cast(self._byzantine_client_weight, tf.float32)
      elif self._client_weighting is not client_weight_lib.ClientWeighting.UNIFORM:
        weights_delta_weight = self._client_weighting({key: self._byzantine_client_weight for key in model_output})
    return optimizer_utils.ClientOutput(weights_delta, weights_delta_weight, model_output, optimizer_output)

  @staticmethod
  def model_to_client_delta_fn(client_optimizer_fn, *, client_weighting=None, use_experimental_simulation_loop=False,
                               byzantine_client_weight: int = 1_000_000):
    """Returns a function that accepts a model creation function and returns a `ClientDeltaFn` instance.

        Args:
          client_optimizer_fn: A no-arg callable that returns a `tf.keras.Optimizer`.
          client_weighting: A value of `tff.learning.ClientWeighting` that specifies a
            built-in weighting method, or a callable that takes the output of
            `model.report_local_outputs` and returns a tensor that provides the weight
            in the federated average of model deltas. If None, defaults to weighting
            by number of examples.
          use_experimental_simulation_loop: Controls the reduce loop function for
              input dataset. An experimental reduce loop is used for simulation.
              It is currently necessary to set this flag to True for performant GPU
              simulations.
          byzantine_client_weight: Number of samples each Byzantine client reports.

        Returns:
          A function that accepts a model creation function and returns a `ClientDeltaFn` instance."""

    def ret(model_fn: Callable[[], model_lib.Model]) -> optimizer_utils.ClientDeltaFn:
      return ByzantineWeightClientFedAvg(model_fn(), client_optimizer_fn(), client_weighting,
                                         use_experimental_simulation_loop, byzantine_client_weight)

    return ret
