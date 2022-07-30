import tensorflow as tf
from attacks.simple_heuristic_omniscient.base import SimpleHeuristicOmniscientAttack


class ScalarPAFAttack(SimpleHeuristicOmniscientAttack):
  def __init__(self, scalar=100.):
    self._scalar = scalar

  def __call__(self, trainable_initial_weights):
    return tf.nest.map_structure(lambda _: _ - _ + self._scalar, trainable_initial_weights)
