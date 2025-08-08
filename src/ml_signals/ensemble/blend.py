import numpy as np
def weighted_average(probs_list, weights=None):
    n = len(probs_list)
    if weights is None: weights = np.ones(n)/n
    weights = np.array(weights); weights = weights/weights.sum()
    return np.stack(probs_list, axis=1).dot(weights)
