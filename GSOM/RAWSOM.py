import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin

class SOM(object):

    def __init__(self, lr = 0.5, n_neighbors = 100, iters = 20):

        self.n_neighbors = n_neighbors
        self.lrst = lr
        self.iters = iters

    def fit_transform(self, X):
        self.train(X)
        return self.predict(X)

    def train(self, X):
        m = 60
        n = 60
        self.W = np.zeros((m*n, X.shape[1]))
        self.Y = np.array([[i, j] for i in range(m) for j in range(n)])
        for i in range(self.iters):
            ntime = i * 1./self.iters
            lr = self.lrst * np.exp(-ntime)
            # self.n_neighbors = self.n_neighbors *0.8
            for x in X:

                bmu = np.linalg.norm(x - self.W, axis=1).argmin()
                ldist = np.linalg.norm(self.Y[bmu] - self.Y, axis=1)
                neighbors = ldist.argsort()[:self.n_neighbors*(1-ntime)]
                r = np.sqrt(self.n_neighbors/2.)
                H = np.array([np.exp(-.5 * (ldist[neighbors]/r)**3)]).T
                self.W[neighbors] += lr * (x - self.W[neighbors]) * H
            print ('\r %i / %i : %.4f' %((i), (self.iters), self.n_neighbors)),

    def predict(self, X):

        out = []

        for x in X:
            out.append(self.Y[np.linalg.norm(x - self.W, axis=1).argmin()])

        return np.array(out)
