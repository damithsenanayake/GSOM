from scipy.optimize import minimize
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances, pairwise_distances_argmin_min
import numpy as np

def QE(X, W):
    bmus,dists = pairwise_distances_argmin_min(X, W)

    return np.sum(dists)


