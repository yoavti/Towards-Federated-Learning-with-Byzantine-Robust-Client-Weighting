import attr

from typing import Optional

from tensorflow_federated.python.aggregators.factory import AggregationFactory

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
    default='shakespeare_character',
    validator=attr.validators.instance_of(str))
  """A string specifying the training task."""
  aggregation_factory: Optional[AggregationFactory] = attr.ib(
    default=None)
  """model_update_aggregation_factory: An optional `tff.aggregators.AggregationFactory`
  that constructs `tff.templates.AggregationProcess` for aggregating the client model updates on the server."""
