from ortools.linear_solver.pywraplp import Solver
import numpy as np
from shared.utils import EPSILON, trunc_helpers


def lp(N, *, alpha=0.1, alpha_star=0.5):
  K, t = trunc_helpers(N, alpha)
  N = np.array(N)
  idx = np.argsort(N)[::-1]
  inv_idx = np.empty_like(idx)
  inv_idx[idx] = np.arange(idx.shape[0])
  N = N[idx]
  alpha_star -= EPSILON  # helps deal with numerical errors

  # Create the linear solver using the CBC backend
  solver = Solver('Minimize L1 distance', Solver.GLOP_LINEAR_PROGRAMMING)

  # 1. Create the variables we want to optimize
  # This definition adds the following implicit constraint: 0 <= n' <= n
  ns = [solver.IntVar(0, int(n), f'n_{i}') for i, n in enumerate(N)]
  xis = [solver.IntVar(0, int(n), f'xi_{i}') for i, n in enumerate(N)]

  # 2. Add constraints for each resource
  # 2.1. mwp
  solver.Add(sum(ns[:t]) <= alpha_star * sum(ns))
  # 2.2. Decreasing Order
  for i in range(K - 1):
    solver.Add(ns[i] >= ns[i + 1])
  # 2.3. xi = |n' - n|
  for i in range(K):
    solver.Add(xis[i] >= N[i] - ns[i])
    solver.Add(xis[i] <= N[i] - ns[i])

  # 3. Minimize the objective function
  solver.Minimize(sum(xis))

  # Solve problem
  solver.Solve()

  ret = [n.solution_value() for n in ns]
  ret = np.array(ret)
  ret = ret[inv_idx]
  return ret
