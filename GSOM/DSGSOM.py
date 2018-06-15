
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin
import timeit


class GSOM(object):

    def __init__(self, n_neighbors=600, lrst=0.1, sf=0.9, fd=0.15, wd=0.02, beta=0):
        self.lrst = lrst
        self.sf = sf
        self.fd = fd
        self.wd = wd
        self.beta = 0
        self.radst = np.sqrt(n_neighbors/2)
        self.n_neighbors = n_neighbors

    def fit_transform(self, X):
        self.train_batch(X)
        return self.predict(X)

    def train_batch(self, X):
        its = 8
        st = timeit.default_timer()
        self.GT = - (X.max()-X.min()) * X.shape[1]* np.log(self.sf)
        self.grid = np.array([[i,j] for i in range(2) for j in range(2)])
        self.W = np.random.random(size=(self.grid.shape[0], X.shape[1]))
        self.errors = np.zeros(self.grid.shape[0])
        for i in range(its):
            self.rad = self.radst# * np.exp(-.5*(i/float(its))**2)
            self.lr = self.lrst* np.exp(-.5 *(i/float(its))**4)
            xix = 0
            if self.rad < 1:
                break
            for x in X:
                xix += 1
                ''' Training For Instances'''
                bmu = pairwise_distances_argmin(np.array([x]), self.W, axis=1)[0]
                ldist = np.linalg.norm(self.grid - self.grid[bmu], axis=1)
                neighbors = np.where(ldist < self.rad)
                hdist = np.linalg.norm(self.W[neighbors] - self.W[bmu], axis=1)
                theta_d = np.array([np.exp(-15.5 * (ldist[neighbors]/self.rad)**2)]).T
                theta_D = np.array([1-np.exp(-45.5*(hdist/hdist.max())**6)]).T
                self.errors[bmu]+= np.linalg.norm(self.W[bmu]-x)
                self.W[neighbors]+= (x-self.W[neighbors])*theta_d*self.lr - theta_D*self.wd*self.W[neighbors]*(np.exp(-8.5*((i-its*0.5)/float(its))**2))
                et = timeit.default_timer()-st
                print ('\riter %i : %i / %i : |G| = %i : radius :%.4f : LR: %.4f  p(g): %.4f Rrad: %.2f'%(i+1,xix, X.shape[0], self.W.shape[0], self.rad, self.lr,  np.exp(-8.*(i/float(its))**2), (self.n_neighbors*1./self.W.shape[0]) )),' time = %.2f'%(et),

                ''' Growing When Necessary '''
                while self.errors.max() >= self.GT :#and np.random.binomial(1, np.exp(-8.*(i/float(its))**2)):
                    # cands = np.where(self.errors >= self.GT)[0]
                    # for g_node in cands:
                    g_node = self.errors.argmax()
                    up = self.grid[g_node] + np.array([0, 1])
                    left = self.grid[g_node] + np.array([-1, 0])
                    down = self.grid[g_node] + np.array([0, -1])
                    right = self.grid[g_node] + np.array([1, 0])

                    imm_neis = [up, left, down, right]

                    for nei in imm_neis:
                        if self.point_exists(self.grid, nei):
                            n_point = self.find_point(self.grid, nei)
                            self.errors[n_point]+=self.errors[n_point]*self.fd
                        else:
                            gdists_new = np.linalg.norm(nei-self.grid, axis=1)
                            gdists_old = np.linalg.norm(self.grid - self.grid[g_node], axis=1)
                            closest2_new = np.argsort(gdists_new)[:2]
                            if np.any(gdists_old[closest2_new]==1):
                                w = self.W[closest2_new].mean(axis=0)
                            else:
                                w = self.W[closest2_new[0]]*2-self.W[closest2_new[1]]

                            self.W = np.append(self.W, np.array([w]), axis=0)
                            self.errors = np.append(self.errors, self.GT/2)
                            self.grid = np.append(self.grid, np.array([nei]), axis=0)
                    self.errors[g_node]=self.GT/2


    def point_exists(self, space, point):
        return not np.linalg.norm(space-point, axis=1).all()

    def find_point(self, space, point):
        return np.where(np.linalg.norm(space-point, axis=1)==0)[0]

    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.grid[np.argmin(np.linalg.norm(x-self.W, axis=1))])
        return np.array(Y)
