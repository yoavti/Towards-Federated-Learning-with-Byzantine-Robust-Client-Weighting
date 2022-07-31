from shared.attacks.spec import AttackSpec
from shared.attacks.base_client_fed_avg import ByzantineClientFedAvg, ByzantineWeightClientFedAvg
from shared.attacks.collusion import COLLUSION_ATTACKS, CollusionAttackClientFedAvg
from shared.attacks.local import LOCAL_ATTACKS, LocalAttackClientFedAvg


def configure_attack(attack_spec: AttackSpec):
  client_optimizer_fn = attack_spec.client_optimizer_fn
  client_weighting = attack_spec.client_weighting
  byzantine_client_weight = attack_spec.byzantine_client_weight
  attack = attack_spec.attack

  if not byzantine_client_weight:
    return ByzantineClientFedAvg.model_to_client_delta_fn(client_optimizer_fn,
                                                          client_weighting=client_weighting)

  if attack in COLLUSION_ATTACKS:
    if not byzantine_client_weight:
      raise ValueError
    attack_fn = COLLUSION_ATTACKS[attack]()
    return CollusionAttackClientFedAvg.model_to_client_delta_fn(client_optimizer_fn,
                                                                client_weighting=client_weighting,
                                                                byzantine_client_weight=byzantine_client_weight,
                                                                attack=attack_fn)
  elif attack in LOCAL_ATTACKS:
    attack_fn = LOCAL_ATTACKS[attack]()
    return LocalAttackClientFedAvg.model_to_client_delta_fn(client_optimizer_fn,
                                                            client_weighting=client_weighting,
                                                            byzantine_client_weight=byzantine_client_weight,
                                                            attack=attack_fn)
  else:
    return ByzantineWeightClientFedAvg.model_to_client_delta_fn(client_optimizer_fn,
                                                                client_weighting=client_weighting,
                                                                byzantine_client_weight=byzantine_client_weight)
