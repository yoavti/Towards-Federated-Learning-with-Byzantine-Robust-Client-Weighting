def create_preprocess_total(preproc_transform):
  def ret(weights):
    return preproc_transform.transform(weights)
  return ret


def create_preprocess_round(preproc_transform):
  def ret(weights):
    return preproc_transform.fit_transform(weights)
  return ret


PREPROCESS_FUNC_CREATORS = {'total': create_preprocess_total, 'round': create_preprocess_round}
