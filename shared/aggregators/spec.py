import attr
from typing import Optional, Callable, Any
from shared.flags_validators import add_exception, create_in_validator


AGGREGATORS = ['aggregator']
WEIGHT_PREPROCS = ['weight_preproc']
BYZANTINES_PART_OF = ['total']


@attr.s(eq=False, order=False, frozen=True)
class AggregatorSpec(object):
  """Contains information for configuring aggregators.

  Attributes:
    aggregation: A string specifying which aggregation method to use.
    preprocess: A function accepting weights and returning preprocessed weights.
  """
  aggregation: str = attr.ib(
    default='aggregator',
    validator=[attr.validators.instance_of(str), add_exception(create_in_validator(AGGREGATORS))])
  preprocess: Optional[Callable[[Any], Any]] = attr.ib(
    default=None,
    validator=[attr.validators.optional(attr.validators.is_callable())])
