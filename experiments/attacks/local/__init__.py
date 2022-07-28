from .constant import ConstantAttack
from .gaussian import GaussianAttack
from .random_sign_flip import RandomSignFlipAttack
from .sign_flip import SignFlipAttack


ATTACKS = {'sign_flip': SignFlipAttack, 'constant': ConstantAttack, 'gaussian': GaussianAttack,
           'random_sign_flip': RandomSignFlipAttack}  # delta_to_zero
