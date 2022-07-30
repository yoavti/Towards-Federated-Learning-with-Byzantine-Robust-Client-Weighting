import tensorflow as tf
from attacks.local.base import LocalAttack


class ConstantAttack(LocalAttack):
  def __init__(self, scalar=100.):
    self._scalar = scalar

  def __call__(self, weights_delta):
    return tf.nest.map_structure(lambda _: _ - _ + self._scalar, weights_delta)
