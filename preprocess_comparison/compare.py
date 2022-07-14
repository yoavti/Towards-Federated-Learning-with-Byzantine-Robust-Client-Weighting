import numpy as np


def l1_metric(v1, v2):
    return np.linalg.norm(v1 - v2, 1)
