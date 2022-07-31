import attr
from typing import Optional, Callable, Any
from shared.flags_validators import add_exception, create_in_validator, check_proportion, create_optional_validator
from shared.aggregators.options import ALL_AGGREGATORS


@attr.s(eq=False, order=False, frozen=True)
class AggregatorSpec(object):
  """Contains information for configuring aggregators."""
  aggregation: str = attr.ib(
    default='aggregator',
    validator=[attr.validators.instance_of(str), add_exception(create_in_validator(ALL_AGGREGATORS))])
  """A string specifying which aggregation method to use."""
  preprocess: Optional[Callable[[Any], Any]] = attr.ib(
    default=None,
    validator=[attr.validators.optional(attr.validators.is_callable())])
  """A function accepting weights and returning preprocessed weights."""
  alpha: Optional[float] = attr.ib(
    default=None,
    validator=[attr.validators.optional(attr.validators.instance_of(float)),
               add_exception(create_optional_validator(check_proportion))])
  """A float specifying Byzantine proportion."""
