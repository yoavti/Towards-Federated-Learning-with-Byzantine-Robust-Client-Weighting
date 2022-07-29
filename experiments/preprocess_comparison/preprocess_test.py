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
from experiments.preprocess_comparison.load import get_client_weights
from shared.preprocess import lp, truncate
from shared.preprocess.utils import maximal_weight_proportion


ALPHA = 0.1
ALPHA_STAR = 0.5
DATASET = 'emnist'
LIMIT_COUNT = 10


class PreprocessTest(unittest.TestCase):
  def _is_valid_solution(self, N):
    mwp = maximal_weight_proportion(N, ALPHA)
    self.assertLessEqual(mwp, ALPHA_STAR, 'mwp of resulting weights is larger that alpha*')

  def _test_valid(self, preprocess):
    N = get_client_weights(DATASET, LIMIT_COUNT)
    N_ = preprocess(N, alpha=ALPHA, alpha_star=ALPHA_STAR)
    self._is_valid_solution(N_)

  def test_lp_valid(self):
    self._test_valid(lp)

  def test_truncate_valid(self):
    self._test_valid(truncate)


if __name__ == '__main__':
  unittest.main()
