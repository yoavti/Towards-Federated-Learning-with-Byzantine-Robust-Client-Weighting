from functools import partial

from shared.aggregators.spec import AggregatorSpec
from shared.aggregators.numpy_aggregation_factory import NumpyAggregationFactory
from shared.aggregators.numpy_aggregators import mean, median, trimmed_mean
from shared.aggregators.options import AGGREGATORS


def _configure_inner_aggregator(aggregation, alpha):
  if aggregation == 'mean':
    return mean
  elif aggregation == 'median':
    return median
  elif aggregation == 'trimmed_mean':
    if not alpha:
      raise ValueError('If aggregation = trimmed_mean, alpha must be specified')
    return partial(trimmed_mean, beta=alpha)
  return mean


def _build_aggregate_with_preprocess(inner_aggregator, preprocess):
  def aggregate_with_preprocess(points, weights):
    weights = preprocess(weights)
    return inner_aggregator(points, weights)
  return aggregate_with_preprocess


def _configure_numpy_aggregator(aggregation, preprocess, alpha):
  if not preprocess and aggregation == 'mean':
    return None
  inner_aggregator = _configure_inner_aggregator(aggregation, alpha)
  if preprocess:
    inner_aggregator = _build_aggregate_with_preprocess(inner_aggregator, preprocess)
  aggregation_factory = NumpyAggregationFactory(inner_aggregator)
  return aggregation_factory


def configure_aggregator(aggregator_spec: AggregatorSpec):
  aggregation = aggregator_spec.aggregation
  preprocess = aggregator_spec.preprocess
  alpha = aggregator_spec.alpha

  if aggregation in AGGREGATORS:
    aggregation_factory = _configure_numpy_aggregator(aggregation, preprocess, alpha)
  else:
    return None
  return aggregation_factory
