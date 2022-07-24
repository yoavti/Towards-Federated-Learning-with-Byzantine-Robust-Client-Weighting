from experiments.attacks.local.base import LocalAttack
import tensorflow as tf


class SignFlipAttack(LocalAttack):
  def attack(self, weights_delta):
    return tf.nest.map_structure(lambda _: -_, weights_delta)
