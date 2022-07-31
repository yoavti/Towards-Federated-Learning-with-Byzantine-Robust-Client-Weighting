import attr
from typing import Optional, Iterable
from shared.flags_validators import add_exception, create_in_validator, check_proportion


BYZANTINES_PART_OF = {'total', 'round'}


@attr.s(eq=False, order=False, frozen=True)
class PreprocessSpec(object):
  """Contains information for configuring attacks."""
  weight_preproc: str = attr.ib(
    default='weight_preproc',
    validator=[attr.validators.instance_of(str)])
  """A string specifying what to do with the clients' relative weights."""
  byzantines_part_of: str = attr.ib(
    default='byzantines_part_of',
    validator=[attr.validators.instance_of(str), add_exception(create_in_validator(BYZANTINES_PART_OF))])
  """A string specifying what whether preprocessing should
    take all clients into account or
    only those in the current round."""
  alpha: float = attr.ib(
    default=0.1,
    validator=[attr.validators.instance_of(float), add_exception(check_proportion)])
  """A float specifying Byzantine proportion."""
  alpha_star: float = attr.ib(
    default=0.1,
    validator=[attr.validators.instance_of(float), add_exception(check_proportion)])
  """A float specifying Byzantine weight proportion."""
  all_weights: Optional[Iterable[int]] = attr.ib(
    default=None)
  """A generator client weights."""
