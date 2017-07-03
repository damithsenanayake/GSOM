import numpy as np
from scipy.spatial.distance import minkowski
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import sys

class GSOM(object):

    def __init__(self, dims, sf, fd, max_nodes, min_nodes, radius = 5):
        self.radius = radius
        self.neurons = {}
        self.grid = {}
        self.errors = {}
        self.hits = {}
        self.w1 = {}
        self.w2 = {}
        self.a = {}
        self.b = {}

        for i in range(2):
            for j in range(2):
                self.neurons[str([i, j])] = 2*np.random.random(dims)-1
                self.grid[str([i, j])] = [i, j]
                self.errors[str([i, j])] = 0

        self.sf = sf
        self.GT = -dims * np.log(sf)#/255.0
        self.fd = fd
        self. max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.dims = dims
        self.range = radius

    def reconstruct(self, X, neu):
        l1 = self.sig(np.dot(np.array([X]), self.w1[neu]) + self.a[neu])  # 1/(1+np.exp(-(np.dot(X,syn0))))
        l2 = self.sig(np.dot(l1, self.w2[neu]) + self.b[neu])
        return l2

    def train(self, X, neu,  iters, eps, momentum=0.5, wd_param=0.1):
        for i in range(iters):
            l1 = self.sig(np.dot(X, self.w1[neu]) + self.a[neu])  # 1/(1+np.exp(-(np.dot(X,syn0))))
            l2 = self.sig(np.dot(l1, self.w2[neu]) + self.b[neu])  # 1/(1+np.exp(-(np.dot(l1,syn1))))
            l2_delta = (X - l2) * (l2 * (1 - l2))
            l1_delta = l2_delta.dot(self.w2[neu].T) * (l1 * (1 - l1))

            # sparsity :
            avg = np.average(l1, axis=0)
            targ = np.ones(avg.shape) * 0.0001
            sparse_term = (- targ / avg + (1 - targ) / (1 - avg)) * (l1 * (1 - l1))
            l1_delta += 0.01 * sparse_term

            dw2 = eps * (l1.T.dot(l2_delta) + momentum * self.mw2 - wd_param * self.w2)
            dw1 = eps * (X.T.dot(l1_delta) + momentum * self.mw1 - wd_param * self.w1)
            da = eps * (l1_delta.sum(axis=0) + momentum * self.ma - wd_param * self.ma)
            db = eps * (l2_delta.sum(axis=0) + momentum * self.mb - wd_param * self.mb)
            self.w2[neu] += dw2
            self.w1[neu] += dw1
            self.a[neu] += da
            self.b[neu] += db
            self.mw1[neu] = dw1
            self.mw2[neu] = dw2
            self.ma[neu] = da
            self.mb[neu] = db
        return l2


    def batch_nn_train(self, X, lr):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=-1).fit(np.array(self.neurons.values()))
        errs, inds = nbrs.kneighbors(X)

        for node in np.unique(inds):
            neighborhood, dists = self.get_neighbourhood(self.neurons.keys()[node])
            hs = np.exp(-(dists**2 / 2*self.range**2))
            xavg = np.average(X[np.where(inds == node)[0]], axis=0)
            erravg = np.average(errs[np.where(inds == node)[0]])
            for neighbor, h in zip(neighborhood, hs):
                self.neurons[neighbor] += (1/h if h > 0 else 1)* lr * (xavg - self.neurons[neighbor])
                try:
                    self.errors[neighbor] += erravg
                except KeyError:
                    self.errors[neighbor] = erravg


    # def train_with_pruning(self, X, iterations, lr, prune_thresh):
    #     self.lr = lr
    #     for i in range(iterations):
    #         c = 0
    #         t = X.shape[0]
    #         for x in X:
    #             c += 1
    #             self.train_single(x)
    #             sys.stdout.write(
    #                 '\r epoch %i / %i :  %i%% : nodes - %i' % (i + 1, iterations, c * 100 / t, len(self.w)))
    #         self.lr *= (1 - 3.8 / len(self.w))
    #         if i > 1 and i%prune_thresh ==0:
    #             self.predict(X)
    #             self.prune_unused()


    def train_batch(self, X, iterations, lr):
        self.lr = lr
        for i in range(iterations):
            c = 0
            lamda = (i+1) / iterations
            t = X.shape[0]
            if True:#len(self.w) < self.max_nodes:
                for x in X:
                    c+=1
                    self.train_single(x)
                    sys.stdout.write('\r epoch %i / %i :  %i%% : nodes - %i' %(i+1, iterations,c*100/t, len(self.neurons) ))
                    sys.stdout.flush()
            else:
                self.batch_nn_train(X, self.lr)
                sys.stdout.write('\r epoch %i / %i : nodes - %i' %((i+1), iterations, len(self.neurons)))
                sys.stdout.flush()
            self.lr *= (1 - 3.8 / len(self.neurons))
            self.range = self.radius * np.exp(- i / iterations)

    def prune_unused(self):
        for k in self.neurons.keys():
            try:
                del self.hits[k]
            except KeyError:
                try:
                    del self.neurons[k]
                    del self.grid[k]
                    del self.errors[k]
                except :
                    continue


    def train_single(self, x):
        bmu, err = self.find_bmu(x)
        neighbors , dists = self.get_neighbourhood(bmu)
        hs = np.exp(-(dists / 2*self.range))

        for neighbor, h in zip(neighbors, hs):
            self.neurons[neighbor] += h * self.lr* (x - self.neurons[neighbor]) #* self.lr * (x - self.w[neighbor])
            # self.w[neighbor] +=  self.lr* (x - self.w[neighbor]) #* self.lr * (x - self.w[neighbor])
            try:
                if self.errors[neighbor] > self.GT and self.max_nodes > len(self.neurons):
                    self.grow(neighbor)
            except:
                continue
        try:
            self.errors[bmu] += err
        except KeyError:
            self.errors[bmu] = err

        # if self.errors[bmu] > self.GT and self.max_nodes > len(self.w):
        #     self.grow(bmu)

    def predict(self, X):
        arr = []
        hits = []
        for x in X:
            hit = self.find_bmu(x)[0]
            hits.append(hit)
            try:
                self.neurons[hit] += 1
            except KeyError:
                self.neurons[hit] = 1
            arr.append(self.grid[hit])

        return np.array(arr), hits


    def find_bmu(self, x):
        nodes = np.asarray(self.neurons.values())
        deltas = nodes - x
        dist_sqr = np.sum(deltas**2, axis =1 )
        mink = np.argmin(dist_sqr)
        # mink = pairwise_distances_argmin(nodes, np.array([x]))
        try:
            dist =minkowski(self.neurons.values()[mink], x, p = 2)
        except ValueError:
            print 'nan'
        return self.neurons.keys()[mink], dist  #dist_sqr[mink]

    def get_neighbourhood(self, node):
        p_dist_matrix = pairwise_distances(np.array(self.grid.values()))
        #np.fill_diagonal(p_dist_matrix, np.Infinity)
        node_dists = p_dist_matrix[np.where(np.array(self.neurons.keys())==node)[0]][0]
        return np.array(self.grid.keys())[np.where(node_dists< self.range)[0]], node_dists[np.where(node_dists<self.range)[0]]


    def grow(self, bmu):
        # type: (object) -> object
        p = self.grid[bmu]
        up = p + np.array([0, -1])
        down = p + np.array([0, +1])
        left = p + np.array([-1, 0])
        right = p + np.array([+1, 0])

        neighbors = np.array([up, right, down, left])
        direction = 0
        for nei in neighbors:
            try:
                self.errors[str(list(nei))] += self.errors[bmu] * self.fd
            except KeyError:
                new_b = self.type_b(nei, direction)
                new_a = self.type_a(nei, direction)
                new_c = self.type_c(nei, direction)

                if new_b.all() == 0:
                    if new_a.all() == 0:
                        if new_c.all() == 0:
                            w = np.random.random(self.dims)-0.5
                            # w.fill(0.5)
                        else:
                            w = new_c
                    else:
                        w = new_a
                else:
                    w = new_b

                self.neurons[str(list(nei))] = w
                self.grid[str(list(nei))] = list(nei)
                self.errors[str(list(nei))] = self.GT/2
            direction += 1
        self.errors[bmu] = self.GT / 2

    def type_b(self, nei, direction):
        try:
            if direction == 0 or direction == 2:
                return (self.neurons[str(list(nei + np.array([0, -1])))]+ self.neurons[
                    str(list((nei + np.array([0, 1]))))]) * 0.5
            return (self.neurons[str(list(nei + np.array([-1, 0])))] + self.neurons[
                str(list(nei + np.array([1, 0])))]) * 0.5
        except KeyError:
            return np.array([0])


    def type_a(self, nei, direction):
        try:
            anc = {0: np.array([0, -1]),
                   1: np.array([1, 0]),
                   2: np.array([0, 1]),
                   3: np.array([-1, 0])}
            w1 = self.neurons[str(list((nei + anc[direction])))]
            w2 = self.neurons[str(list(nei + 2 * anc[direction]))]
            return 2 * w1 - w2
        except KeyError:
            return np.array([0])


    def type_c(self, nei, direction):
        try:
            anc = {0: np.array([0, -1]),
                   1: np.array([1, 0]),
                   2: np.array([0, 1]),
                   3: np.array([-1, 0])}

            if direction == 0 or direction == 2:
                try:
                    return 2 * self.neurons[str(list(nei + anc[direction]))] - self.neurons[
                        str(list(nei + anc[direction] + np.array([1, 0])))]
                except KeyError:
                    return 2 * self.neurons[str(list(nei + anc[direction]))] - self.neurons[
                        str(list(nei + anc[direction] + np.array([-1, 0])))]

            else:
                try:
                    return 2 * self.neurons[str(list(nei + anc[direction]))] - self.neurons[
                        str(list(nei + anc[direction] + np.array([0, 1])))]
                except KeyError:
                    return 2 * self.neurons[str(list(nei + anc[direction]))] - self.neurons[
                        str(list(nei + anc[direction] + np.array([0, -1])))]
        except KeyError:
            return np.array([0])

