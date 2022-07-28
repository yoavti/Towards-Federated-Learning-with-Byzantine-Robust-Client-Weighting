# Copyright 2022, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from experiments.attacks.local import ConstantAttack, GaussianAttack, NoAttack, RandomSignFlipAttack, SignFlipAttack


ATTACKS = {'none': NoAttack,
           'sign_flip': SignFlipAttack, 'constant': ConstantAttack,
           'gaussian': GaussianAttack,
           'random_sign_flip': RandomSignFlipAttack}


def gen_rand_vec(dn=10):
  return np.random.randn(dn)


class LocalAttacksTest(unittest.TestCase):
  def test_no_attack(self):
    x = gen_rand_vec()
    np.testing.assert_array_almost_equal(x, NoAttack()(x))

  def test_sign_flip(self):
    x = gen_rand_vec()
    np.testing.assert_array_almost_equal(-x, SignFlipAttack()(x))

  def test_constant(self):
    const = 100
    x = gen_rand_vec()
    np.testing.assert_array_almost_equal(const, ConstantAttack(const)(x))


if __name__ == '__main__':
  unittest.main()
