from functools import partial
from shared.aggregators.spec import AggregatorSpec
from shared.aggregators.factory.numpy_aggregation_factory import NumpyAggregationFactory
from shared.aggregators.numpy_aggregators import mean, median, trimmed_mean
from shared.aggregators.factory.robust_weiszfeld_factory import RobustWeiszfeldFactory
from shared.aggregators.factory.preprocessed_aggregation_factory import PreprocessedAggregationFactory
from shared.aggregators.options import NUMPY_AGGREGATORS


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


def _configure_numpy_aggregator(aggregation, preprocess, alpha):
  if not preprocess and aggregation == 'mean':
    return None
  inner_aggregator = _configure_inner_aggregator(aggregation, alpha)
  aggregation_factory = NumpyAggregationFactory(inner_aggregator)
  return aggregation_factory


def configure_aggregator(aggregator_spec: AggregatorSpec):
  aggregation = aggregator_spec.aggregation
  preprocess = aggregator_spec.preprocess
  alpha = aggregator_spec.alpha

  if aggregation in NUMPY_AGGREGATORS:
    aggregation_factory = _configure_numpy_aggregator(aggregation, preprocess, alpha)
  elif aggregation == 'rsa':
    aggregation_factory = RobustWeiszfeldFactory()
  else:
    return None
  if preprocess:
    aggregation_factory = PreprocessedAggregationFactory(aggregation_factory, preprocess)
  return aggregation_factory
