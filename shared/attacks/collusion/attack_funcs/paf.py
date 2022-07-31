import tensorflow as tf
from shared.attacks.collusion.attack_funcs.base import CollusionAttack


class ScalarPAFAttack(CollusionAttack):
  def __init__(self, scalar=100.):
    self._scalar = scalar

  def __call__(self, trainable_initial_weights):
    return tf.nest.map_structure(lambda _: _ + self._scalar, trainable_initial_weights)
