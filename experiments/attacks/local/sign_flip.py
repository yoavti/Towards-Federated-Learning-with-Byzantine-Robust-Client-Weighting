from experiments.attacks.local.base import LocalAttack
import tensorflow as tf


class SignFlipAttack(LocalAttack):
  def __call__(self, weights_delta):
    return tf.nest.map_structure(lambda _: -_, weights_delta)
