import numpy as np
from math import isclose
from preprocess.utils import trunc_helpers, maximal_weight_proportion, is_valid_solution, Preprocess


def _trunc(vec, threshold):
  return np.where(vec > threshold, threshold, vec)


def _find_U(N, *, alpha=0.1, alpha_star=0.5):
  N = np.array(N)
  N = np.sort(N)
  N = N[::-1]
  K, t = trunc_helpers(N, alpha)
  for u, n_u in enumerate(N):
    truncated = _trunc(N, n_u)
    if not is_valid_solution(truncated, alpha, alpha_star):
      continue
    mwp = maximal_weight_proportion(truncated, alpha)
    if isclose(mwp, alpha_star):
      return n_u
    c = N[u:].sum()
    d = u
    if u < t:
      a = N[u:t].sum()
      b = u
    else:
      a = 0
      b = t
    numerator = a - c * alpha_star
    denominator = d * alpha_star - b
    if isclose(denominator, 0):
      return n_u
    return numerator / denominator
  return N[-1]


class Truncate(Preprocess):
  def __init__(self, *, alpha=0.1, alpha_star=0.5):
    super().__init__(alpha=alpha, alpha_star=alpha_star)
    self.U = 0

  def fit(self, N):
    self.U = _find_U(N, alpha=self._alpha, alpha_star=self._alpha_star)

  def transform(self, N):
    return _trunc(N, self.U)
