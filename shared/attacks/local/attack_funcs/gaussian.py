import random

import tensorflow as tf

from shared.attacks.local.attack_funcs.base import LocalAttack


class GaussianAttack(LocalAttack):
  def __init__(self, mu=0, sigma=200):
    self._mu = mu
    self._sigma = sigma

  def __call__(self, weights_delta):
    return tf.nest.map_structure(lambda _: _ - _ + random.gauss(self._mu, self._sigma), weights_delta)
