import tensorflow as tf
from tensorflow_federated.python.learning import ClientWeighting
from shared.client_weighting.spec import ClientWeightingSpec
from shared.google_tff_research.utils import task_utils


def configure_client_weight(client_weight_spec: ClientWeightingSpec):
  client_weighting = client_weight_spec.client_weighting
  task = client_weight_spec.task
  if client_weighting is ClientWeighting.NUM_EXAMPLES and task in task_utils.TASKS_NUM_TOKENS:
    def client_weight_fn(local_outputs):
      return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)
  else:
    client_weight_fn = client_weighting
  return client_weight_fn
