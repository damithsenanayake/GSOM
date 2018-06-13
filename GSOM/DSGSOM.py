
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin, pairwise_distances_argmin_min


class GSOM(object):

    def __init__(self, n_neighbors, lrst=0.1, sf=0.9, fd=0.15, wd=0.02, beta=0):
        self.lrst = lrst
        self.sf = sf
        self.fd = fd
        self.wd = wd
        self.beta = 0
        self.radst = np.sqrt(n_neighbors/2)

    def train_batch(self, X):
        its = 10
        self.GT = - (X.max()-X.min()) * X.shape[1]* np.log(self.sf)
        self.grid = np.array([[i,j] for i in range(2) for j in range(2)])
        self.W = np.random.random(size=(self.grid.shape[0], X.shape[1]))
        self.errors = np.zeros(self.grid.shape[0])
        for i in range(its):
            self.rad = self.radst * np.exp(-0.5*(i/float(its))**2)
            self.lr = self.lrst
            for x in range(X):
                ''' Training For Instances'''
                bmu = pairwise_distances_argmin(x, self.W, axis=1)[0]
                ldist = np.linalg.norm(self.grid - self.grid[bmu], axis=1)
                neighbors = np.where(ldist < self.rad)
                hdist = np.linalg.norm(self.W[neighbors] - self.W[bmu], axis=1)
                theta_d = np.array([np.exp(-15.5 * (ldist[neighbors]/self.rad)**0)]).T
                theta_D = np.array([2-np.exp(-.5*(hdist/hdist.max())**2)]).T
                self.W[neighbors]+= (x-self.W[neighbors])*theta_d*self.lr - theta_D*self.wd*self.W[neighbors]*(np.exp(-4.5*(i/float(its))**3))
                self.errors[bmu]+= np.linalg.norm(self.W[bmu]-x)

                ''' Growing When Necessary '''
                while self.errors.max() > self.GT:
                    g_node = self.errors.argmax()
                    up = self.grid[g_node] + np.array([0, 1])
                    left = self.grid[g_node] + np.array([-1, 0])
                    down = self.grid[g_node] + np.array([0, -1])
                    right = self.grid[g_node] + np.array([1, 0])

                    imm_neis = [up, left, down, right]

                    for nei in imm_neis:
                        if self.point_exists(self.grid, nei):
                            n_point = self.grid[self.find_point(self.grid, nei)]
                            self.errors[n_point]*=(1+self.fd)
                        else:
                            gdists = np.linalg.norm(nei-self.grid, axis=1)
                            closest2 = np.argsort(gdists)[:2]

                            if gdists[closest2[0]==closest2[1]]:
                                w = self.W[closest2].sum(axis=0)/2
                            else:
                                w = self.W[closest2[0]]*2-self.W[closest2[1]]

                            self.W = np.append(self.W, np.array([w]), axis=0)
                            self.errors = np.append(self.errors, [0])
                            self.grid = np.append(self.grid, np.array([nei]), axis=0)

    def point_exists(self, space, point):
        return not np.linalg.norm(space-point, axis=1).all()

    def find_point(self, space, point):
        return np.where(np.linalg.norm(space-point, axis=1)==0)[0]
