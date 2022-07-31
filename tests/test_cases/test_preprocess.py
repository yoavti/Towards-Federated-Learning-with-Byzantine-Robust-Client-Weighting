import unittest

import numpy as np

from shared.extract_client_weights import get_client_weights
from shared.preprocess import LP, Truncate
from shared.preprocess.utils import maximal_weight_proportion


ALPHA = 0.1
ALPHA_STAR = 0.5
DATASET = 'emnist'
LIMIT_COUNT = 10


class PreprocessTest(unittest.TestCase):
  def _is_valid_solution(self, N):
    mwp = maximal_weight_proportion(N, ALPHA)
    self.assertLessEqual(mwp, ALPHA_STAR, 'mwp of resulting weights is larger that alpha*')

  def _test_valid(self, preprocess_constructor):
    N = get_client_weights(DATASET, LIMIT_COUNT)
    N = list(N)
    N = np.array(N)
    preprocess_transform = preprocess_constructor(alpha=ALPHA, alpha_star=ALPHA_STAR)
    N_ = preprocess_transform.fit_transform(N)
    self._is_valid_solution(N_)

  def test_lp_valid(self):
    self._test_valid(LP)

  def test_truncate_valid(self):
    self._test_valid(Truncate)


if __name__ == '__main__':
  unittest.main()
