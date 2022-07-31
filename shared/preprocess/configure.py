import numpy as np

from shared.preprocess.spec import PreprocessSpec
from shared.preprocess.methods.dict import PREPROC_TRANSFORMS
from shared.preprocess.utils.preprocess_func_creators import PREPROCESS_FUNC_CREATORS


def configure_preprocess(preprocess_spec: PreprocessSpec):
  weight_preproc = preprocess_spec.weight_preproc
  byzantines_part_of = preprocess_spec.byzantines_part_of
  alpha = preprocess_spec.alpha
  alpha_star = preprocess_spec.alpha_star
  all_weights = preprocess_spec.all_weights

  if weight_preproc not in PREPROC_TRANSFORMS:
    return None
  preproc_transform = PREPROC_TRANSFORMS[weight_preproc](alpha=alpha, alpha_star=alpha_star)
  if byzantines_part_of == 'total':
    if not all_weights:
      raise ValueError('If byzantines_part_of = total, client_datasets must be specified')
    all_weights = list(all_weights)
    all_weights = np.array(all_weights)
    preproc_transform.fit(all_weights)
  preprocess_func = PREPROCESS_FUNC_CREATORS[byzantines_part_of](preproc_transform)
  return preprocess_func
