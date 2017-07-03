import numpy as np
import sys
import scipy.stats as st

from scipy.spatial import Delaunay, ConvexHull
class MovingMap(object):

    def __init__(self, alpha = 0.1, beta= 1,  iterations = 100, delta=0):
        self.alpha = alpha
        self.iterations = iterations
        self.beta = beta
        self.delta = delta
        self.x_cs = []
        self.y_cs =[]

    def get_neighbors(self, point):
        if self.tri_change and (not np.random.binomial(1, self.ts * 1.0 / (
            self.ts + self.fs)) or self.G == None):  # self.ts*1.0 / (self.ts + self.fs)<0.75:

            self.G = self.triangulate()
        if np.all(self.G == self.P):
            self.ts += 1
        else:
            self.fs += 1
            self.P = self.G
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
            adj[line[0]][line[1]] = 0
            adj[line[1]][line[0]] = 0

        return adj


    def fit(self, X):
        batchwise=False
        if X.size > 10000:
            b_size = X.shape[0] /500
            n_batches = 500
            batchwise = True


        gs = np.ceil(np.sqrt(X.shape[0])).astype(int)
        self.grid =np.array([(j , i) for j in range(gs)for i in range(gs)]).astype(float)
        self.grid /= self.grid.max()
        self.C = np.random.random( (self.grid.shape[0],X.shape[1])) *0+ X.mean(axis=0)

        radius = 1
        if True:#not batchwise:
            for iter in range(self.iterations):
                sys.stdout.write('\r iteration %i'% iter)
                radius = np.exp(-6.0*iter/self.iterations)
                for x in X[np.random.permutation(X.shape[0])]:
                    Hdist = np.linalg.norm(x-self.C,axis=1)
                    bmu = np.argmin(Hdist)
                    Ldist = np.linalg.norm(self.grid - self.grid[bmu], axis = 1)
                    neighbors = np.where(Ldist<radius)[0]

                    self.C[neighbors] += self.alpha * (x - self.C[neighbors]) * np.exp(-2.5*iter/self.iterations) * np.array([np.exp(-0.5*Ldist[neighbors]**2/radius**2)]).T
            radius =0.05
            for iter in range(self.iterations):
                sys.stdout.write('\r iteration %i' % iter)
                radius = 0.1*np.exp(-2.0* iter  / self.iterations)
                for i in range(self.grid.shape[0]):
                    Ldist = np.linalg.norm(self.grid - self.grid[i], axis=1)
                    Hdist = np.linalg.norm(self.C[i] - self.C, axis=1)

                    # neighbors = np.where(Ldist<radius)[0]
                    neighbors = np.argsort(Ldist)[:100]
                    try:
                        d = Ldist[neighbors] / Ldist[neighbors].max()
                        d[np.isnan(d)] = 0.0
                        D = Hdist[neighbors] / Hdist[neighbors].max()
                        D[np.isnan(D)] = 0.0
                    except:
                        continue
                    dirs = np.array([np.exp(- D)]).T
                    # ams = np.array([np.exp(-d)])

                    z = (Hdist[neighbors] )/np.sqrt(Hdist[neighbors].var())
                    ps = 2*st.norm.pdf(z)
                    ps[np.isnan(ps)]=0
                    coefs = np.random.binomial(1, ps)
                    try:
                        # coefs[coefs==0]= -1
                        coefs[coefs==1]=1
                    except:
                        continue
                    self.grid[neighbors] += self.beta * np.exp(-2.5 * iter  / self.iterations ) * dirs * (
                        self.grid[i] - self.grid[neighbors])* np.array([coefs]).T# * ams


                    if np.isnan(self.grid).any() or np.isinf(self.grid).any():
                        print 'error '
                    self.grid -= self.grid.min()
                    self.grid /= self.grid.max()

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
