import attr
from shared.client_weighting.dict import CLIENT_WEIGHTING


def _convert_client_weighting(client_weighting):
  return CLIENT_WEIGHTING.get(client_weighting, None)


@attr.s(eq=False, order=False, frozen=True)
class ClientWeightingSpec(object):
  """Contains information for configuring attacks."""
  client_weighting: str = attr.ib(
    default='num_examples',
    converter=_convert_client_weighting)
  """A string specifying the training task."""
  task: str = attr.ib(
    default='shakespeare_character')
  """A string specifying the training task."""
