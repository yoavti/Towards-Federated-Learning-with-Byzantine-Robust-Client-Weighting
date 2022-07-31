from shared.attacks.local.attack_funcs.constant import ConstantAttack
from shared.attacks.local.attack_funcs.gaussian import GaussianAttack
from shared.attacks.local.attack_funcs.random_sign_flip import RandomSignFlipAttack
from shared.attacks.local.attack_funcs.sign_flip import SignFlipAttack


LOCAL_ATTACKS = {'constant': ConstantAttack, 'gaussian': GaussianAttack, 'random_sign_flip': RandomSignFlipAttack,
                 'sign_flip': SignFlipAttack}
