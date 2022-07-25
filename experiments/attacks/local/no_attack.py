from experiments.attacks.local.base import LocalAttack


class NoAttack(LocalAttack):
  def __call__(self, weights_delta):
    return weights_delta
