
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
import scipy.stats as st
import sys

class SelfOrganizingSwarm(object):

    def __init__(self, dim_out=2, verbose=0,iterations=10, alpha=0.5, beta=0.25, delta=0.25, theta=2.05):
        self.alpha = alpha
        self.theta = theta
        self.beta = beta
        self.delta = delta
        self.d = dim_out
        self.verbose = verbose
        self.grid = []
        self.iterations = iterations
        self.G = None
        self.P = None
        self.ts = 0
        self.fs = 0.00000001
        self.neighbors = None

    def get_n_nearest_neighbors(self, point, n):

        return np.argsort(np.linalg.norm(self.grid[point]-self.grid, axis=1))[:n]

    def get_nearest_neighbors(self, point, r):

        neis = np.where(np.linalg.norm(self.grid[point] - self.grid, axis=1) < r)[0]

        return neis  # np.where(self.G[point])[0]

    def get_neighbors(self, point):
        self.G = self.triangulate()
        return np.where(self.G[point])[0]
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
        for line in chull:
            adj[line[0]][line[1]]=0
            adj[line[1]][line[0]]=0

        return adj

    def fit(self, X):
        for i in range(int(np.ceil(np.sqrt(X.shape[0])))):
            for j in range(int(np.ceil(np.sqrt(X.shape[0])))):
                self.grid.append([i, j])
        #
        self.grid = np.array(self.grid).astype(float)
        self.grid /= self.grid.max()
        #
        # self.grid = np.random.randn(X.shape[0]  , self.d)


        # self.grid = np.random.random((X.shape[0]  , self.d))
        self.C = np.ones((self.grid.shape[0], X.shape[1])) * X.mean(axis=0)
        verbosity_limit = self.iterations * self.verbose
        for it in range(self.iterations):
            # if not (it % verbosity_limit):
            sys.stdout.write('\r iteration %s' % str(it + 1))

            self.tri_change = True
            for x in X[np.random.permutation(X.shape[0])]:
                # find BMU
                bmu = np.argmin(np.linalg.norm(self.C - x, axis=1))
                n = np.floor(X.shape[0] * 0.25*np.exp(-0.5*it**2 / self.iterations**2))
                neighbors = self.get_neighbors(bmu)
                # neighbors = self.get_nearest_neighbors(bmu, np.exp(-it/self.iterations ))
                self.tri_change = False
                self.neighbors = neighbors
                dists = np.linalg.norm(self.grid[neighbors] - self.grid[bmu], axis=1)
                if  dists.shape[0] and not dists.max() == 0:
                    rad = dists.max()
                else :
                    continue
                moving_amounts =  np.array([np.exp(-0.5*dists**2/rad**2)]).T#/np.sum(np.exp(-dists**2/rad**2)) #/((np.sum(np.array([np.exp(-dists)]).T) ) * dists.shape[0])
                moving_amounts/=moving_amounts.sum()
                self.C[neighbors] +=self.alpha * (x - self.C[neighbors]) * np.exp(-it**2 / (2*self.iterations**2))  * moving_amounts
                if np.isnan(self.C).any() or np.isinf(self.C).any():
                    print 'error'
            self.tri_change = True
            for x in X:

                i = np.argmin(np.linalg.norm(self.C - x,axis=1))

                neighbors = self.get_neighbors(i)#, 50)#neighbors(i)

                dists = np.linalg.norm(self.C[neighbors] - x, axis=1)

                v = np.sqrt(dists.var())
                m = 0#dists.mean()

                z = (dists)/v
                ps = 1-2*st.norm.pdf(z)
                ps[np.where(np.isnan(ps))] = 0
                ninds = np.where(np.random.binomial(1,ps))[0]#np.where(z < self.theta)[0]
                neis = neighbors[ninds]
                others = np.setdiff1d(np.array(range(X.shape[0])), neis)
                D = dists[ninds]
                D/=D.sum()

                d = np.linalg.norm(self.grid[neis]-self.grid[i], axis=1)
                if d.sum():
                    d /= d.sum()


                difs = d-D
                try:
                    difs /= np.abs(difs).max()
                except:
                    continue
                # D/=D.max()
                indirs = (self.grid[i] - self.grid[neis])
                # indirs /= np.array([np.linalg.norm(indirs, axis=1)]).T
                # indirs[np.where(np.isnan(indirs))[0]]=0
                self.grid[neis] += self.beta* indirs * np.exp(-it/self.iterations)*np.array([np.exp(-D)]).T#* np.array([difs]).T

                d_ = np.linalg.norm(self.C[others] - self.C[i], axis = 1)
                if d_.shape[0] and d_.max():
                    d_/=d_.max()

                dirs = self.grid[i] - self.grid[others]
                l = np.array([np.linalg.norm(dirs, axis = 1)]).T
                l[l==0]=1
                dirs/=l

                self.grid[others] -= self.delta*np.array([np.exp(-d_)]).T*dirs* np.exp(-it/self.iterations)

                if np.isnan(self.grid).any() or np.isinf(self.grid).any():
                    print 'error'

                # self.grid[i] -= self.beta *(np.array([np.exp(-D**2)]).T* (self.grid[i] - self.grid[neis]) * np.exp(-it/self.iterations)).sum(axis=0)
                # sys.stdout.write('\r %i'%(i+1))
                # sys.stdout.flush()

            # for x in X:
            #     bmu = np.argmin(np.linalg.norm(self.C - x, axis=1))
            #     neighbors = self.get_neighbors(bmu)
            #     self.neighbors = neighbors
            #     others = np.setdiff1d(np.array(range(X.shape[0])), neighbors)
            #
            #     dists = np.linalg.norm(self.C[neighbors] - x, axis=1)
            #     dist_ld = np.linalg.norm(self.grid[neighbors] - self.grid[bmu], axis = 1)
            #     mean = dists.mean()
            #     variance = dists.var()
            #     mean_ld = dist_ld.mean()
            #     variance_ld = dist_ld.var()
            #
            #
            #     try:
            #         rad = dists.max()
            #         rad_ld = dist_ld.max()
            #     except:
            #         continue
            #     inds = np.where((dists - mean)/np.sqrt(variance) >self.theta)[0]
            #     far_ones = neighbors[inds]
            #     close_ones = np.setdiff1d(neighbors, far_ones)
            #     nei_inds = np.setdiff1d(np.array(range(dists.shape[0])), inds)
            #     try:
            #         nei_dists =dists[nei_inds]
            #         D = nei_dists/nei_dists.sum()
            #         nei_dist_ld = dist_ld[nei_inds]
            #         d = nei_dist_ld / nei_dist_ld.sum()
            #     except:
            #         continue
            #     neighbors = close_ones
            #     not_neighbors = far_ones
            #     moving_amounts = np.array([np.exp(-0.5*nei_dists**2/rad**2)]).T# * (1- np.exp(-np.square(d-D) ))]).T#* (nei_dist_ld/rad_ld)]).T #/ nei_dists.shape[0]
            #     moving_amounts /= moving_amounts.sum()
            #     moving_directions = self.grid[bmu] - self.grid[neighbors]
            #     moving_directions /= np.array([np.linalg.norm(moving_directions, axis=1)]).T
            #     moving_directions[np.isnan(moving_directions)] = 0
            #     self.grid[neighbors] += self.beta*np.exp(-it / (0.5 * self.iterations)) * moving_amounts * moving_directions
            #
            #     not_nei_dists = dists[inds]
            #     not_nei_dists_ld = dist_ld[inds]
            #     if not not_nei_dists.shape[0] == 0:
            #         not_nei_dists/=rad
            #         not_nei_dists_ld /= rad_ld
            #     moving_amounts =np.array([np.exp(-0.5*not_nei_dists**2)]).T#* (not_nei_dists_ld)]).T#/ not_nei_dists.shape[0]
            #     moving_directions = self.grid[bmu] - self.grid[not_neighbors]
            #     moving_directions /= np.array([np.linalg.norm(moving_directions, axis=1)]).T
            #
            #     self.grid[not_neighbors] -= self.delta*np.exp(-it**2 / (2  * self.iterations**2)) * moving_amounts * moving_directions

            #     others = np.setdiff1d(np.array(range(X.shape[0])), neighbors)
            #
            #     dists = np.linalg.norm(self.C[others] - x, axis=1)
            #     dists /= (dists.max())
            #     moving_directions = self.grid[others]-self.grid[bmu]
            #     moving_amounts = np.array([np.exp(-0.5* dists ** 2)]).T
            #
            #     self.grid[others] -= self.delta*np.exp(-it**2/(2*self.iterations**2)) * moving_amounts * moving_directions


    def predict(self, X):
        out = []
        for x in X:
            bmu = np.argmin(np.linalg.norm(self.C - x, axis=1))
            out.append(self.grid[bmu])
        out = np.array(out)
        return out


    def fit_transform(self, X):
        self.fit(X)
        return self.predict(X)