import numpy as np
from preprocess.utils.transforms.preprocess_transform import Preprocess


class VectorMap(Preprocess):
  def __init__(self, *, alpha=0.1, alpha_star=0.5):
    super().__init__(alpha=alpha, alpha_star=alpha_star)
    self._vector_map = {}

  def _vector_computation(self, N):
    raise NotImplementedError

  def fit(self, N):
    self._vector_map = {}
    N_ = self._vector_computation(N)
    for n, n_ in zip(N, N_):
      if n not in self._vector_map:
        self._vector_map[n] = []
      self._vector_map[n].append(n_)
    self._vector_map = {n: np.array(arr) for n, arr in self._vector_map.items()}

  def transform(self, N):
    return np.vectorize(lambda n: self._vector_map.get(n, 0))(np.array(N))
