import tensorflow as tf
from tensorflow_federated.python.learning import ClientWeighting
from tensorflow_federated.python.aggregators.factory import UnweightedAggregationFactory
from shared.client_weighting.spec import ClientWeightingSpec
from shared.google_tff_research.utils import task_utils


def configure_client_weight(client_weight_spec: ClientWeightingSpec):
  client_weighting = client_weight_spec.client_weighting
  task = client_weight_spec.task
  aggregation_factory = client_weight_spec.aggregation_factory

  if client_weighting is ClientWeighting.NUM_EXAMPLES and task in task_utils.TASKS_NUM_TOKENS:
    def client_weight_fn(local_outputs):
      return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)
  else:
    client_weight_fn = client_weighting

  if isinstance(aggregation_factory, UnweightedAggregationFactory) and client_weighting is not ClientWeighting.UNIFORM:
    raise ValueError('Cannot use non-uniform client weighting with unweighted aggregation.')
  return client_weight_fn
