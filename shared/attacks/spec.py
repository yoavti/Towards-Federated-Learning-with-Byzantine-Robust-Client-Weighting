import attr
from shared.flags_validators import add_exception, create_in_validator
from shared.attacks.dict import ALL_ATTACKS


@attr.s(eq=False, order=False, frozen=True)
class AttackSpec(object):
  """Contains information for configuring attacks."""
  attack: str = attr.ib(
    default='sign_flip',
    validator=[attr.validators.instance_of(str), add_exception(create_in_validator(ALL_ATTACKS))])
  """A string specifying which attack takes place."""
