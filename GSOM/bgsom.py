import numpy as np
from scipy.spatial.distance import minkowski
from sklearn.metrics.pairwise import pairwise_distances
import timeit

import sys

class GSOM(object):

    def __init__(self):
        self.radius = 0
        self.lr = 0
        self.neurons = {}
        self.grid = {}
        self.errors = {}
        self.hits = {}
        self.gen = {}
        self.cache = []

        self.means={}


        self.p_dist_matrix = None
        self.grid_changed = True
        self.sf = 0
        self.GT = 0
        self.fd = 0

        self.dims = 0
        self.range = 0
        self.Herr = 0
        self.current_gen = 0

    def fit(self, X, sf , fd,  lr, beta ):
        self.dims = X.shape[1]

        self.radius = np.exp(1)#3.0#
        self.beta = beta
        for i in range(2):
            for j in range(2):
                self.neurons[str([i, j])] = np.random.random(self.dims)
                self.grid[str([i, j])] = [i, j]
                self.errors[str([i, j])] = 0
        self.sf = sf
        self.GT = -self.dims * np.log(sf)  # /255.0
        self.fd = fd
        self.Herr = 0

        self.lr = lr
        st = timeit.default_timer()
        self.train_batch(X[np.random.permutation(np.array(range(X.shape[0])))])
        et = timeit.default_timer() - st
        print "\n elapsed time for growing : ", et , "\n"
        self.Y = np.array(self.grid.values()).astype(float)
        self.Y -= self.Y.min(axis=0)
        self.Y /= self.Y.max(axis=0)
        self.C = np.array(self.neurons.values())
        self.smoothen(X)


    def smoothen(self, X, lr = 0.5):
        r_st =0.9
        its =100
        print self.wd
        st = timeit.default_timer()
        Ydists = pairwise_distances(self.Y)
        for i in range(its):
            radius =r_st* np.exp(-2.0*i/its)
            alpha =lr -i * lr * 1.0 / its #* np.exp(-1.5*i/(its))
            sys.stdout.write('\r smoothing epoch %i: %s : radius: %s' % (i+1, str(alpha), str(radius)))
            sys.stdout.flush()
            for x in X:

                bmu = np.argmin(np.linalg.norm(x - self.C, axis=1))
                Ldist = Ydists[bmu] #np.linalg.norm(self.Y-self.Y[bmu], axis = 1)
                neighborhood =np.where(Ldist < radius)[0]
                # neighborhood =np.argsort(Ldist)[:5]

                w = np.array(self.C)[neighborhood]
                w +=  alpha * ((x-w) * np.array([np.exp(-(15.5)*Ldist[neighborhood]**2/radius**2)]).T- self.wd*-w*(1-np.exp(-2.5*i/its)))
                if np.any(np.isinf(w)) or np.any(np.isnan(w)):
                    print 'error'
                self.C[neighborhood] = w
        print "\ntime for first smoothing iteration : ", (timeit.default_timer()-st), "\n"


    def predict_inner(self, X):
        hits = []
        for x in X:
            hit = np.argmin(np.linalg.norm(x - self.C, axis=1))
            hits.append(hit)

        return np.array(hits)

    def fit_transform(self, X ,sf = 0.1, fd=0.5,  lr = 1, beta=0.1 , wd = 0.04):
        self.wd = wd
        self.fit(X, sf = sf, fd = fd, lr = lr, beta=beta)
        return self.LMDS(X)

    def train_batch(self, X):
        i = 0
        lr = self.lr
        while self.lr > 0.5:
            c = 0
            t = X.shape[0]
            self.Herr=0

            for x in X:
                c+=1
                self.train_single(x)
                sys.stdout.write('\r epoch %i :  %i%% : nodes - %i : LR - %s : radius : %s' %(i+1, c*100/t, len(self.neurons), str(self.lr), str(self.radius) ))
                sys.stdout.flush()
            self.lr *=0.9* (1 - 3.8 / len(self.neurons))# np.exp(-i/50.0)#
            self.radius *=np.exp(-i/200.0)#(1 - 3.8 / len(self.w))
            if self.radius <=1:
                break#self.radius = 1.1
            # if self.Herr > self.GT:
            for k in self.errors.keys():
                self.errors[k] = 0
            i += 1



    def train_single(self, x):
        bmu, err = self.find_bmu(x)
        neighbors , dists = self.get_neighbourhood(bmu)
        # self.fd = 1.0 / len(neighbors)
        hs = np.array([np.exp(-dists**2/(2*self.radius**2))]).T
        # hs.fill(1)
        weights = np.array(self.neurons.values())[neighbors]

        weights +=  (x - weights) * self.lr*hs #- self.lr * self.wd*weights

        for neighbor, w in zip(np.array(self.neurons.keys())[neighbors], weights):
            self.neurons[neighbor] = w
        try:
            self.errors[bmu] += err
        except KeyError:
            self.errors[bmu] = err
        if self.errors[bmu] > self.Herr:
                self.Herr = self.errors[bmu]

        if self.errors[bmu] > self.GT:
        #if self.errors[bmu] > self.GT:
            self.grow(bmu)
            for k in np.array(self.neurons.keys())[neighbors]:
                if not k == bmu:
                    self.errors[k]+= (self.errors[bmu]*self.fd)

    def predict(self, X):
        hits =[]
        for x in X:
            hit = self.Y[np.argmin(np.linalg.norm(x-self.C, axis = 1))]
            hits.append(hit)

        return np.array(hits)


    def find_bmu(self, x):
        nodes = np.asarray(self.neurons.values())
        mink = np.argmin(np.linalg.norm(x - nodes, axis=1))
        # mink = pairwise_distances_argmin(nodes, np.array([x]))
        try:
            dist =minkowski(self.neurons.values()[mink], x, p = 2)
        except ValueError:
            print 'nan'

        return self.neurons.keys()[mink], dist   #dist_sqr[mink]

    def get_neighbourhood(self, node):
        if self.grid_changed:
            self.p_dist_matrix = pairwise_distances(np.array(self.grid.values()))
            self.grid_changed=False
        node_dists = self.p_dist_matrix[np.where(np.array(self.neurons.keys()) == node)[0]][0]
        return np.where(node_dists < self.radius)[0], node_dists[
            np.where(node_dists < self.radius)[0]]  # np.array(self.grid.keys())



    def grow(self, bmu):
        # type: (object) -> object
        p = self.grid[bmu]
        up = p + np.array([0, +1])
        right = p + np.array([+1, 0])
        down = p + np.array([0, -1])
        left = p + np.array([-1, 0])

        neighbors = np.array([up, right, down, left])
        for nei in neighbors:
            try:
                self.errors[str(list(nei))] = self.errors[str(list(nei))] * 1.0
            except KeyError:
                self.grid_changed=True
                w = self.get_new_weight(bmu, nei)

                self.neurons[str(list(nei))] = w
                self.grid[str(list(nei))] = list(nei)
                self.errors[str(list(nei))] =(self.errors[bmu] -self.GT/2)* self.fd

        self.errors[bmu] = self.GT / 2


    def get_new_weight(self, old, new):

        grid = np.array(self.grid.values())
        dists = np.linalg.norm(grid - np.array(new), axis = 1)
        order1 = np.where(dists ==1)[0]
        order2 = np.where(dists==2)[0]
        order2L = np.where(dists==np.sqrt(2))[0]
        w1 = self.neurons[old]
        # if order1.shape[0] > 1:
        try :
            w2 = self.neurons[self.grid.keys()[order1[np.where(np.linalg.norm(np.array(self.grid.values())[order1] - np.array(self.grid[old]), axis=1)==2)[0]]]]
            return (w1+w2)/2
        except TypeError:
            second_neighbours = order2[np.where(np.linalg.norm(np.array(self.grid.values())[order2] - np.array(self.grid[old]), axis=1) == 1)[0]]
            third_neighbours = order2L[np.where(np.linalg.norm(np.array(self.grid.values())[order2L] - np.array(self.grid[old]), axis=1) == 1)[0]]
            try:
                w2 = self.neurons[self.grid.keys()[second_neighbours[0]]]
            except:
                try:
                    w2 = self.neurons[self.grid.keys()[third_neighbours[0]]]
                except:
                    w2 = np.random.random(self.dims) * 2 -1
            return 2 * w1 - w2

########################################################################################################################

    def LMDS(self, X):
        r_st = .3
        radius = r_st

        grid = self.predict(X).astype(float)
        n = X.shape[0]*0.5
        its = 70
        it = 0
        st = timeit.default_timer()

        while it < its and radius > 0.001 and self.beta*np.exp(-7.5 * it**2  / its**2 ) > 0.001:# or n>1:
            radius *=0.9# r_st *  np.exp(- 9.0*it  / (50))

            sys.stdout.write('\r LMDS iteration %i : radius : %s : beta : %s' % (it, str(radius), str(self.beta *np.exp(-7.5 * it**2  / its**2 ))))
              # np.exp(-10.0* iter  / self.iterations)
            for i in range(X.shape[0]):
                grid -= grid.min(axis=0)
                grid /= grid.max(axis=0)
                Ldist = np.linalg.norm(grid - grid[i], axis=1)
                Hdist = np.linalg.norm(X[i] - X, axis=1)

                neighbors = np.where(Ldist < radius)[0]
                # neighbors = np.argsort(Ldist)[:10]
                if len(neighbors.shape) == 0 or neighbors.shape[0] == 1 or not Ldist[neighbors].any():
                    continue
                d = Ldist[neighbors] / Ldist[neighbors].sum()
                d[np.isnan(d)] = 0
                D = Hdist[neighbors] / Hdist[neighbors].sum()
                D[np.isnan(D)] = 0
                dirs = np.array([d - D]).T
                try:
                    ds = d/d.max()
                    hs = np.exp(-0.5 * ds**2)
                    hs = np.array([hs]).T
                except:
                    break
                if len(neighbors) == 0:
                    pass
                grid[neighbors] += self.beta  * np.exp(-7.5 * it**2  / its**2 ) * dirs * (
                grid[i] - grid[neighbors]) * hs


                # if np.isnan(self.grid).any() or np.isinf(self.grid).any():
                #     print 'error '
            it += 1
            n*=0.8
        print '\n LMDS time : ', timeit.default_timer() - st
        print len(self.neurons)
        return grid

