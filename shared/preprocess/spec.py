import attr
from shared.flags_validators import add_exception, create_in_validator


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
  """
  weight_preproc: str = attr.ib(
    default='weight_preproc',
    validator=[attr.validators.instance_of(str), add_exception(create_in_validator(WEIGHT_PREPROCS))])
  byzantines_part_of: str = attr.ib(
    default='byzantines_part_of',
    validator=[attr.validators.instance_of(str), add_exception(create_in_validator(BYZANTINES_PART_OF))])
