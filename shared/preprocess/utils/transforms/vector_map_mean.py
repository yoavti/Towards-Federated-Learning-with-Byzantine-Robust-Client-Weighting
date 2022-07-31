import numpy as np
from shared.preprocess.utils.transforms.vector_map import VectorMap


class VectorMapMean(VectorMap):
  def __init__(self, *, alpha=0.1, alpha_star=0.5):
    super().__init__(alpha=alpha, alpha_star=alpha_star)
    self._vector_map = {}

  def _vector_computation(self, N):
    raise NotImplementedError

  def fit(self, N):
    super().fit(N)
    self._vector_map = {n: np.mean(arr) for n, arr in self._vector_map.items()}
