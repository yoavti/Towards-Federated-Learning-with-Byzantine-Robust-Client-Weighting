EPSILON = 1e-3


def trunc_helpers(N, alpha):
    N = list(N)
    K = len(N)
    t = int(K * alpha + 1)
    return K, t
