
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from sklearn.neighbors import NearestNeighbors
import sys

class SelfOrganizingSwarm(object):

    def __init__(self, dim_out=2, verbose=0,iterations=1000,alpha = 1,  beta = 0.1):
        self.d = dim_out
        self.verbose = verbose
        self.grid = []
        self.iterations = iterations
        self.G = None
        self.P = None
        self.ts = 0
        self.fs = 0.00000001
        self.neighbors = None
        self.beta = beta
        self.alpha = alpha
        self.degree = 1
        self.deg_max = self.degree

    def get_immediate_neighbors(self, point):
        try:
            if not np.random.binomial(1, self.ts * 1.0 / (
                self.ts + self.fs)) or self.G == None:  # self.ts*1.0 / (self.ts + self.fs)<0.75:
                self.G = self.triangulate()
                # for i in range(max(int(self.degree), 1
                #                    )):
                #     self.G += self.G.dot(self.G.T)
                # # self.degree = (self.degree- (self.degree*1.0/self.iterations))
                # self.G[self.G > 0] = 1
        except ValueError:
            print "hold"
        if np.all(self.G == self.P):
            self.ts += 1
        else:
            self.fs += 1
            self.P = self.G

        return np.where(self.G[point])[0]


    def get_neighbors(self, point, r):

        neis = np.where(np.linalg.norm(self.grid[point]-self.grid, axis=1)<r)[0]

        return neis#np.where(self.G[point])[0]
        # return np.linalg.norm(self.grid - self.grid[point], axis=1).argsort()[:2]

    def triangulate(self):
        adj = np.zeros((self.grid.shape[0], self.grid.shape[0]))
        tri = Delaunay(self.grid)
        chull = ConvexHull(self.grid).simplices
        #     print chull
        for simplegrid in tri.simplices:
            for vert in simplegrid:
                for vert2 in simplegrid:
                    adj[vert][vert2] = 1
        # for line in chull:
        #     adj[line[0]][line[1]]=0
        #     adj[line[1]][line[0]]=0

        return adj

    def fit(self, X):
        for i in range(int(np.ceil(np.sqrt(X.shape[0])))):
            for j in range(int(np.ceil(np.sqrt(X.shape[0])))):
                self.grid.append([i, j])

        self.grid = np.array(self.grid).astype(float)
        self.grid_length = int(np.ceil(np.sqrt(X.shape[0])))

        self.grid = np.random.random((X.shape[0] , self.d))
        self.C = np.ones((self.grid.shape[0], X.shape[1])) * X.mean()
        verbosity_limit = self.iterations * self.verbose
        for it in range(self.iterations):
            # if not (it % verbosity_limit):
            sys.stdout.write('\r iteration %s' % str(it + 1))
            for x in X[np.random.permutation(X.shape[0])]:
                # find BMU
                bmu = np.argmin(np.linalg.norm(self.C - x, axis=1))
                neighbors = self.get_neighbors(bmu, 0.25*np.exp(-0.5*it**2 / self.iterations**2))
                self.neighbors = neighbors
                dists = np.linalg.norm(self.grid[neighbors] - self.grid[bmu], axis=1)
                if not dists.shape[0]:
                    continue
                rad = dists.max()
                moving_amounts =  np.array([np.exp(-0.5*dists**2/rad**2)]).T#/np.sum(np.exp(-dists**2/rad**2)) #/((np.sum(np.array([np.exp(-dists)]).T) ) * dists.shape[0])
                # moving_amounts/=moving_amounts.sum()
                self.C[neighbors] += self.alpha* (x - self.C[neighbors]) * np.exp(-0.5*it**2 / (self.iterations)**2) * moving_amounts


            # nbrs = NearestNeighbors(n_neighbors=100).fit(self.grid)
            # dists, neighbors = nbrs.kneighbors(self.grid)
        # self.degree = self.deg_max
        print 'moving'
        for it in range(self.iterations):
            sys.stdout.write('\r iteration %s' % str(it + 1))

            for i in range(self.C.shape[0]):

                # neis = self.get_neighbors(i,self.grid_length* 0.75*np.exp(-0.5*it**2/self.iterations**2))#neighbors[i]
                neis = self.get_immediate_neighbors(i)
                D = np.linalg.norm(self.C - self.C[i], axis=1)[neis]
                if D.sum():
                    D/= D.sum()
                d = np.linalg.norm(self.grid - self.grid[i], axis=1)[neis]
                if d.sum():
                    d /= d.sum()

                diffs = np.array([D - d]).T
                if np.sum(np.abs(D-d)):
                    diffs /= np.sum(np.abs(D - d))

                if np.isnan(d).any() or np.isnan(D).any() or np.isnan(diffs).any():
                    print "error 1"
                self.grid[neis] -= 0.7 * np.exp(-it /self.iterations )*diffs * (self.grid[i] - self.grid[neis])#* np.array([np.exp(-2*d**2)]).T#
                if np.isnan(self.grid).any() or np.isinf(self.grid).any():
                    print "error"
                    return





    def predict(self, X):
        out = []
        for x in X:
            bmu = np.argmin(np.linalg.norm(self.C - x, axis=1))
            out.append(self.grid[bmu])
        out = np.array(out)/np.abs(np.array(out)).min()
        return out


    def fit_transform(self, X):
        self.fit(X)
        return self.predict(X)