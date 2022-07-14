EPSILON = 1e-3


def trunc_helpers(N, alpha):
    N = list(N)
    K = len(N)
    t = int(K * alpha + 1)
    return K, t


def maximal_weight_proportion(N, alpha):
  K, t = trunc_helpers(N, alpha)
  return N[:t].sum() / N.sum()


def is_valid_solution(N, alpha, alpha_star):
  mwp = maximal_weight_proportion(N, alpha)
  return mwp <= alpha_star
