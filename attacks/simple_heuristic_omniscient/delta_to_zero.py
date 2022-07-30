import tensorflow as tf
from attacks.simple_heuristic_omniscient.base import SimpleHeuristicOmniscientAttack


class DeltaToZeroAttack(SimpleHeuristicOmniscientAttack):
  def __call__(self, trainable_initial_weights):
    return tf.nest.map_structure(lambda _: -_, trainable_initial_weights)
