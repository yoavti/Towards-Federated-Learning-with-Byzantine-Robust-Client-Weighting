import tensorflow as tf
from attacks.local.base import LocalAttack


class SignFlipAttack(LocalAttack):
  def __call__(self, weights_delta):
    return tf.nest.map_structure(lambda _: -_, weights_delta)
