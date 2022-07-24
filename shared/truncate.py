from math import isclose
import numpy as np
from shared.utils import EPSILON, trunc_helpers, maximal_weight_proportion, is_valid_solution


def trunc(vec, threshold):
  return np.where(vec > threshold, threshold, vec)


def find_U(N, *, alpha_star=0.5, alpha=0.1):
  N = np.array(N)
  N = np.sort(N)
  N = N[::-1]
  K, t = trunc_helpers(N, alpha)
  alpha_star -= EPSILON  # helps deal with numerical errors
  for u, n_u in enumerate(N):
    truncated = trunc(N, n_u)
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


def truncate(N, *, alpha_star=0.5, alpha=0.1):
  U = find_U(N, alpha_star=alpha_star, alpha=alpha)
  return trunc(N, U)
