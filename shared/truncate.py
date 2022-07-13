from itertools import islice
from math import ceil, isclose, floor
import numpy as np


def find_alpha(U, N, alpha_star=0.5, sort_N=False):
    if sort_N:
        N = sorted(N, reverse=True)
    # given an upper bound on each n find the
    # maximal number of Byzantine workers that have less than alpha_star weight proportion
    truncated = [min(U, n_k) for n_k in N]
    allowed_weight = sum(truncated) * alpha_star
    top_k_total_weight = 0
    for k, k_weight in enumerate(truncated, start=1):
        top_k_total_weight += k_weight
        if top_k_total_weight > allowed_weight:
            return (k - 1) / len(N)
            # len(N) * alpha


EPSILON = 1e-3


def trunc_helpers(N, alpha):
    N = list(N)
    K = len(N)
    t = int(K * alpha + 1)
    return K, t


def maximal_weight_proportion(N, alpha):
  K, t = trunc_helpers(N, alpha)
  return N[:t].sum() / N.sum()


def trunc(vec, threshold):
    return np.where(vec > threshold, threshold, vec)


def is_valid_solution(N, alpha, alpha_star):
  mwp = maximal_weight_proportion(N, alpha)
  return mwp <= alpha_star


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
        return numerator / denominator
    return N[-1]


def find_U_alpha_pairs(N, alpha_star=0.5):
    # N should be sorted in reverse
    N = sorted(N, reverse=True)

    return [(U, find_alpha(U, N, alpha_star)) for U in list(dict.fromkeys(N))], alpha_star


def plot_U_alpha_pairs(U_alpha_pairs, alpha_star):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib.offsetbox import AnchoredText

    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots()

    ax.set_xlabel(r'$\alpha$')#, rotation=0)
    ax.set_ylabel(r'$U$', rotation=0, labelpad=8)
    ax.xaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    ax.plot(*zip(*U_alpha_pairs))

    ax.add_artist(AnchoredText(f'$\\alpha^*={int(alpha_star * 100)}\\%$', loc='upper right'))

    plt.show()
    fig.savefig('tradeoff.pdf', format='pdf')


def find_U_alpha_pairs_2(N, alpha_star=0.5):
    # N should be sorted in reverse
    K = len(N)
    N = sorted(N)

    f_star = ceil(K * alpha_star)
    alpha_star = f_star / K  # actual percent considering we don't have half clients

    f = f_star
    alpha = alpha_star

    res = []
    while f != 0:
        U = find_U(N, alpha=alpha, alpha_star=alpha_star)

        res.append((alpha, U))

        f -= 1
        alpha = f / K
    return res

