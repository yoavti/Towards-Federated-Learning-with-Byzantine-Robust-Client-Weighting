import unittest

import numpy as np

from shared.attacks.collusion import DeltaToZeroAttack, ScalarPAFAttack
from shared.attacks.local import ConstantAttack, SignFlipAttack


def gen_rand_vec(dn=10):
  return np.random.randn(dn)


class CollusionAttacksTest(unittest.TestCase):
  def test_delta_to_zero(self):
    v = gen_rand_vec()
    np.testing.assert_array_almost_equal(-v, DeltaToZeroAttack()(v))

  def test_paf(self):
    const = 100
    v = gen_rand_vec()
    np.testing.assert_array_almost_equal(v + const, ScalarPAFAttack(const)(v))


class LocalAttacksTest(unittest.TestCase):
  def test_constant(self):
    const = 100
    v = gen_rand_vec()
    np.testing.assert_array_almost_equal(np.full_like(v, const), ConstantAttack(const)(v))

  def test_sign_flip(self):
    v = gen_rand_vec()
    np.testing.assert_array_almost_equal(-v, SignFlipAttack()(v))


if __name__ == '__main__':
  unittest.main()
