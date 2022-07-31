import attr

from typing import Callable, Optional, Any, Union

import tensorflow as tf

from tensorflow_federated.python.learning.client_weight_lib import ClientWeightType, ClientWeighting


@attr.s(eq=False, order=False, frozen=True)
class AttackSpec(object):
  """Contains information for configuring attacks."""
  client_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer] = attr.ib(
    validator=attr.validators.is_callable())
  """A no-arg callable that returns a `tf.keras.Optimizer`."""
  client_weighting: Union[ClientWeightType, Callable[[Any], tf.Tensor]] = attr.ib(
    default=ClientWeighting.NUM_EXAMPLES)
  """A value of `tff.learning.ClientWeighting` that specifies a built-in weighting method,
    or a callable that takes the output of `model.report_local_outputs`
    and returns a tensor that provides the weight in the federated average of model deltas.
    If None, defaults to weighting by number of examples."""
  byzantine_client_weight: Optional[int] = attr.ib(
    default=None,
    validator=attr.validators.optional(attr.validators.instance_of(int)))
  """Number of samples each Byzantine client reports."""
  attack: str = attr.ib(
    default='sign_flip',
    validator=[attr.validators.instance_of(str)])
  """A string specifying which attack takes place."""
