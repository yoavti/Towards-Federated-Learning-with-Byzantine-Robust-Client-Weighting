from experiments.attacks.local.base import LocalAttack


class NoAttack(LocalAttack):
  def attack(self, weights_delta):
    return weights_delta
