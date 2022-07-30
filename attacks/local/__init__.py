from attacks.local.constant import ConstantAttack
from attacks.local.gaussian import GaussianAttack
from attacks.local.random_sign_flip import RandomSignFlipAttack
from attacks.local.sign_flip import SignFlipAttack


ATTACKS = {'constant': ConstantAttack, 'gaussian': GaussianAttack, 'random_sign_flip': RandomSignFlipAttack,
           'sign_flip': SignFlipAttack}  # delta_to_zero
