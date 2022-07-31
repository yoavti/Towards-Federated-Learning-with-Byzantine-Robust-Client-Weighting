import tensorflow as tf

from shared.attacks.collusion.attack_funcs.base import CollusionAttack


class DeltaToZeroAttack(CollusionAttack):
  def __call__(self, trainable_initial_weights):
    return tf.nest.map_structure(lambda _: -_, trainable_initial_weights)
