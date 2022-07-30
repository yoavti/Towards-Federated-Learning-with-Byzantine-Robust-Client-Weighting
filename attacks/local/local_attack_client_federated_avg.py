from tff_patch.federated_averaging import ByzantineWeightClientFedAvg

from typing import Callable, Optional

import tensorflow as tf

from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model as model_lib

from tff_patch import optimizer_utils
from attacks.local.attack_funcs.base import LocalAttack


class LocalAttackClientFedAvg(ByzantineWeightClientFedAvg):
  """Client TensorFlow logic for Federated Averaging with Byzantine clients using local attacks."""

  def __init__(self, model, optimizer, client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
               use_experimental_simulation_loop=False, byzantine_client_weight=1_000_000,
               attack: Optional[LocalAttack] = None):
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
                  attack: An optional `LocalAttack` that specifies which Byzantine attack takes place.
                """
    super().__init__(model, optimizer, client_weighting, use_experimental_simulation_loop, byzantine_client_weight)
    self._attack = attack

  @tf.function
  def __call__(self, dataset_with_byzflag, initial_weights):
    client_output = super().__call__(dataset_with_byzflag, initial_weights)
    weights_delta = client_output.weights_delta
    weights_delta_weight = client_output.weights_delta_weight
    model_output = client_output.model_output
    optimizer_output = client_output.optimizer_output
    dataset, byzflag = dataset_with_byzflag

    if byzflag and self._attack:
      weights_delta = self._attack(weights_delta)
    return optimizer_utils.ClientOutput(weights_delta, weights_delta_weight, model_output, optimizer_output)

  @staticmethod
  def model_to_client_delta_fn(client_optimizer_fn, *, client_weighting=None, use_experimental_simulation_loop=False,
                               byzantine_client_weight=1_000_000,
                               attack: Optional[LocalAttack] = None):
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
          attack: An optional `LocalAttack` that specifies which Byzantine attack takes place.

        Returns:
          A function that accepts a model creation function and returns a `ClientDeltaFn` instance."""

    def ret(model_fn: Callable[[], model_lib.Model]) -> optimizer_utils.ClientDeltaFn:
      return LocalAttackClientFedAvg(model_fn(), client_optimizer_fn(), client_weighting,
                                     use_experimental_simulation_loop, byzantine_client_weight, attack)

    return ret
