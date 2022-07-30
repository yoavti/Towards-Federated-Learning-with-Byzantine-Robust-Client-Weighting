EPSILON = 1e-3


class Preprocess:
  def __init__(self, *, alpha=0.1, alpha_star=0.5):
    self._alpha = alpha
    self._alpha_star = alpha_star
    self._alpha_star -= EPSILON  # helps deal with numerical errors

  def fit(self, N):
    raise NotImplementedError

  def transform(self, N):
    raise NotImplementedError

  def fit_transform(self, N):
    self.fit(N)
    return self.transform(N)
