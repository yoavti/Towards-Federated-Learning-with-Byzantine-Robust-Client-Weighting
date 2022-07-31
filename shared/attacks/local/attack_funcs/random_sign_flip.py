import random

import tensorflow as tf

from shared.attacks.local.attack_funcs.base import LocalAttack


class RandomSignFlipAttack(LocalAttack):
  def __init__(self, mu=-2, sigma=1):
    self._mu = mu
    self._sigma = sigma

  def __call__(self, weights_delta):
    noise = random.gauss(self._mu, self._sigma)
    return tf.nest.map_structure(lambda _: _ * noise, weights_delta)
