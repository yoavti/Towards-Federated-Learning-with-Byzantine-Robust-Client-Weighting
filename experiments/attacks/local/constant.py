from experiments.attacks.local.base import LocalAttack
import tensorflow as tf


class ConstantAttack(LocalAttack):
  def __init__(self, scalar=100.):
    self._scalar = scalar

  def __call__(self, weights_delta):
    return tf.nest.map_structure(lambda _: _ - _ + self._scalar, weights_delta)
