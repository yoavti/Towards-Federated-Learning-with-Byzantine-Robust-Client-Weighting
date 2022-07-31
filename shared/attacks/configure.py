from shared.attacks.spec import AttackSpec
from shared.attacks.collusion import COLLUSION_ATTACKS
from shared.attacks.local import LOCAL_ATTACKS


def configure_attack(attack_spec: AttackSpec):
  attack = attack_spec.attack
  if attack in COLLUSION_ATTACKS:
    return COLLUSION_ATTACKS[attack]()
  elif attack in LOCAL_ATTACKS:
    return LOCAL_ATTACKS[attack]()
  else:
    return None
