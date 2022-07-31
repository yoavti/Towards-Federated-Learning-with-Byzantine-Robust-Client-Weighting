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
