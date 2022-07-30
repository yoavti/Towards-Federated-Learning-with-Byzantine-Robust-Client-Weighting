from tff_patch.federated_averaging import ByzantineWeightClientFedAvg

from typing import Any, Callable, Optional, Union

import tensorflow as tf

from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model as model_lib

from tff_patch import optimizer_utils
from attacks.local.base import LocalAttack


class LocalAttackClientFedAvg(ByzantineWeightClientFedAvg):
  """Client TensorFlow logic for Federated Averaging with Byzantine clients using local attacks."""

  def __init__(self, model: model_lib.Model, optimizer: tf.keras.optimizers.Optimizer,
               client_weighting: Union[client_weight_lib.ClientWeightType,
                                       Callable[[Any], tf.Tensor]] = client_weight_lib.ClientWeighting.NUM_EXAMPLES,
               use_experimental_simulation_loop: bool = False, byzantine_client_weight: int = 1_000_000,
               attack: Optional[LocalAttack] = None):

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
