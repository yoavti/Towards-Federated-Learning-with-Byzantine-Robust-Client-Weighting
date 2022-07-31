import attr
from shared.flags_validators import add_exception, create_in_validator


ATTACKS = ['local']


@attr.s(eq=False, order=False, frozen=True)
class AttackSpec(object):
  """Contains information for configuring attacks."""
  attack: str = attr.ib(
    default='local',
    validator=[attr.validators.instance_of(str), add_exception(create_in_validator(ATTACKS))])
  """A string specifying which attack takes place."""
