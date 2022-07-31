import attr
import numpy as np
from typing import Optional
from shared.flags_validators import add_exception, create_in_validator, check_proportion


WEIGHT_PREPROCS = ['weight_preproc']
BYZANTINES_PART_OF = ['total']


@attr.s(eq=False, order=False, frozen=True)
class PreprocessSpec(object):
  """Contains information for configuring attacks.

  Attributes:
    weight_preproc: A string specifying what to do with the clients' relative weights.
    byzantines_part_of: A string specifying what whether preprocessing should
      take all clients into account or
      only those in the current round.
    alpha: A float specifying Byzantine proportion.
    alpha_star: A float specifying Byzantine weight proportion.
    all_weights: An optional ndarray containing the weights of all clients
  """
  weight_preproc: str = attr.ib(
    default='weight_preproc',
    validator=[attr.validators.instance_of(str), add_exception(create_in_validator(WEIGHT_PREPROCS))])
  byzantines_part_of: str = attr.ib(
    default='byzantines_part_of',
    validator=[attr.validators.instance_of(str), add_exception(create_in_validator(BYZANTINES_PART_OF))])
  alpha: float = attr.ib(
    default=0.1,
    validator=[attr.validators.instance_of(float), add_exception(check_proportion)])
  alpha_star: float = attr.ib(
    default=0.1,
    validator=[attr.validators.instance_of(float), add_exception(check_proportion)])
  all_weights: Optional[np.ndarray] = attr.ib(
    default=None,
    validator=attr.validators.optional(attr.validators.instance_of(np.ndarray)))
