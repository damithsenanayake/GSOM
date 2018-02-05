import numpy as np
import sys
import scipy.stats as st

from scipy.spatial import Delaunay, ConvexHull
class MovingMap(object):

    def __init__(self, alpha = 1, beta= 1,  iterations = 100, delta=0):
        self.alpha = alpha
        self.iterations = iterations
        self.beta = beta
        self.delta = delta
        self.x_cs = []
        self.y_cs =[]



    def fit(self, X):
        batchwise=False
        if X.size > 10000:
            b_size = X.shape[0] /500
            n_batches = 500
            batchwise = True


        gs = np.ceil(np.sqrt(X.shape[0])).astype(int)*2
        self.grid =np.array([(j , i) for j in range(gs)for i in range(gs)]).astype(float)
        self.grid /= self.grid.max()
        self.C = np.random.random( (self.grid.shape[0],X.shape[1])) *0+ X.mean(axis=0)

        radius = 0.25
        if True:#not batchwise:
            for iter in range(self.iterations):
                sys.stdout.write('\r iteration %i'% iter)
                radius = np.exp(-3.0*iter/self.iterations)
                for x in X[np.random.permutation(X.shape[0])]:
                    Hdist = np.linalg.norm(x-self.C,axis=1)
                    bmu = np.argmin(Hdist)
                    Ldist = np.linalg.norm(self.grid - self.grid[bmu], axis = 1)
                    neighbors = np.where(Ldist<radius)[0]

                    self.C[neighbors] += self.alpha * (x - self.C[neighbors]) * np.exp(-2.5*iter/self.iterations) * np.array([np.exp(-0.5*Ldist[neighbors]**2/radius**2)]).T
            radius =1
            for iter in range(self.iterations):
                sys.stdout.write('\r iteration %i' % iter)
                radius *=0.8 #np.exp(-10.0* iter  / self.iterations)
                for i in range(self.grid.shape[0]):
                    Ldist = np.linalg.norm(self.grid - self.grid[i], axis=1)
                    Hdist = np.linalg.norm(self.C[i] - self.C, axis=1)

                    neighbors = np.where(Ldist < radius)[0]
                    # neighbors = np.argsort(Ldist)[:5]
                    if len(neighbors.shape) == 0:
                        continue
                    d = Ldist[neighbors] / Ldist[neighbors].sum()
                    d[np.isnan(d)] = 0
                    D = Hdist[neighbors] / Hdist[neighbors].sum()
                    D[np.isnan(D)] = 0
                    dirs = np.array([d - D]).T
                    # ams = np.array([np.exp(-d)])

                    z = (Hdist[neighbors] - Hdist[neighbors].mean())/np.sqrt(Hdist[neighbors].var())
                    ps = 1-st.norm.cdf(z)
                    ps[np.isnan(ps)]=0
                    coefs = np.random.binomial(1, ps)

                    blastoffs = np.where(coefs ==0)[0]

                    if len(neighbors) == 0:
                        pass
                    self.grid[neighbors] += self.beta * np.exp(-7.5 * iter**2  / self.iterations**2 ) * dirs * (
                        self.grid[i] - self.grid[neighbors])
                    # (self.grid[neighbors])[blastoffs] -= np.exp(-2.5 * iter  / self.iterations ) * (
                    #     self.grid[i] - self.grid[neighbors])[blastoffs]

                    if np.isnan(self.grid).any() or np.isinf(self.grid).any():
                        print 'error '
                    self.grid -= self.grid.min()
                    self.grid /= self.grid.max()
                # radius*= 0.5
            # radius = 1
            # for iter in range(self.iterations):
            #     sys.stdout.write('\r iteration %i'% iter)
            #     radius = np.exp(-2.5*iter/self.iterations)
            #     for i in range(self.grid.shape[0]):
            #         Hdist = np.linalg.norm(self.C[i] - self.C, axis=1)
            #         Ldist = np.linalg.norm(self.grid - self.grid[i], axis=1)
            #
            #         neighbors = np.where(Ldist < radius)[0]
            #
            #         d = Ldist[neighbors]/Ldist[neighbors].sum()
            #         d[np.isnan(d)]=0
            #         D = Hdist[neighbors]/Hdist[neighbors].sum()
            #         D[np.isnan(D)]=0
            #         dirs = np.array([d-D]).T
            #         self.grid[i] -= np.sum(self.delta*np.exp(-2.5* iter/self.iterations)*dirs * (self.grid[i] - self.grid[neighbors]), axis=0)
            #         if np.isnan(self.grid).any() or np.isinf(self.grid).any():
            #             print 'error '
            #         self.grid -= self.grid.min()
            #         self.grid/=self.grid.max()



            #
            # n = 10

            # for iter in range(self.iterations):
            #     sys.stdout.write('\r iteration %i' % iter)
            #     for i in range(self.grid.shape[0]):
            #         Ldist = np.linalg.norm(self.grid - self.grid[i], axis=1)
            #         Hdist = np.linalg.norm(self.C[i] - self.C, axis=1)
            #
            #         neighbors = Ldist.argsort()[:np.floor(n+np.exp(-0.5*iter**2/self.iterations**2)).astype(int)]
            #         d = Ldist[neighbors] / Ldist[neighbors].max()
            #         d[np.isnan(d)] = 0
            #         D = Hdist[neighbors] / Hdist[neighbors].max()
            #         D[np.isnan(D)] = 0
            #         dirs = np.array([d - D]).T
            #
            #         self.grid[neighbors] += self.beta * np.exp(-0.0001 * iter ** 2 / self.iterations ** 2) * dirs * (
            #         self.grid[i] - self.grid[neighbors])
            #         if np.isnan(self.grid).any() or np.isinf(self.grid).any():
            #             print 'error '
            #         self.grid -= self.grid.min()
            #         self.grid /= self.grid.max()




        else:

            for b in range(n_batches):
                sys.stdout.write('\r batch %i\n' % b)

                for iter in range(self.iterations):
                    sys.stdout.write('\r iteration %i' % iter)
                    radius *= np.exp(-0.5 * iter ** 2 / self.iterations ** 2)
                    for x in (X[b*b_size:(b+1)*b_size])[np.random.permutation(b_size)]:
                        Hdist = np.linalg.norm(x - self.C, axis=1)
                        bmu = np.argmin(Hdist)
                        Ldist = np.linalg.norm(self.grid - self.grid[bmu], axis=1)
                        neighbors = np.where(Ldist < radius)[0]

                        self.C[neighbors] += self.alpha * (x - self.C[neighbors]) * np.exp(
                            -0.5 * iter ** 2 / self.iterations ** 2) * np.array(
                            [np.exp(-0.5 * Ldist[neighbors] ** 2 / radius ** 2)]).T

                radius = 1 * np.exp(-0.5*b**2/n_batches**2)

                for iter in range(self.iterations):
                    sys.stdout.write('\r iteration %i' % iter)
                    radius *= np.exp(-0.5 * iter ** 2 / self.iterations ** 2)
                    for i in range(self.grid.shape[0]):
                        Ldist = np.linalg.norm(self.grid - self.grid[i], axis=1)
                        Hdist = np.linalg.norm(self.C[i] - self.C, axis=1)

                        neighbors = np.where(Ldist < radius)[0]

                        d = Ldist[neighbors] / Ldist[neighbors].max()
                        d[np.isnan(d)] = 0
                        D = Hdist[neighbors] / Hdist[neighbors].max()
                        D[np.isnan(D)] = 0
                        dirs = np.array([d - D]).T

                        self.grid[neighbors] += self.beta * np.exp(-0.5 * iter ** 2 / self.iterations ** 2) * dirs * (
                        self.grid[i] - self.grid[neighbors])
                        if np.isnan(self.grid).any() or np.isinf(self.grid).any():
                            print 'error '
                self.grid/=self.grid.max()


    def predict(self, X):
        preds = []

        for x in X:
            preds.append(self.grid[np.argmin(np.linalg.norm(x-self.C, axis=1))])

        return np.array(preds)

    def fit_transform(self, X):
        self.fit(X)
        return self.predict(X)
