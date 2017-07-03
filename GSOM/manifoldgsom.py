import numpy as np
from scipy.spatial.distance import minkowski
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale
import sys

class GSOM(object):

    def __init__(self, dims, sf, fd, max_nodes, min_nodes, radius = 5, lr = 1):
        self.radius = radius
        self.lr = lr
        self.neurons = {}
        self.grid = {}
        self.errors = {}
        self.hits = {}
        self.gen = {}
        self.means={}
        self.visited = None
        for i in range(2):
            for j in range(2):
                self.neurons[str([i, j])] = np.random.random(dims)
                self.grid[str([i, j])] = [i, j]
                self.errors[str([i, j])] = 0
                self.gen[str([i, j])] = 0
                self.hits[str([i, j])] = 0
                self.means[str([i, j])] = 0

        self.sf = sf
        self.GT = -dims * np.log(sf)#/255.0
        self.fd = fd
        self. max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.dims = dims
        self.range = radius
        self.Herr = 0
        self.current_gen = 0

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

    def prune(self):
        for k in self.neurons.keys():
            if  (self.hits[k] < 1 or self.gen[k] < self.current_gen * 0.75):
                del self.gen[k]
                del self.errors[k]
                del self.neurons[k]
                del self.grid[k]
                del self.hits[k]

    def train_batch(self, X, iterations, lr, prune = True):
        self.visited = []
        self.X = X
        if not lr == 0:
            self.lr = lr
        for i in range(iterations):
            c = 0
            lamda = (i+1) / iterations
            t = X.shape[0]
            if True:#len(self.w) < self.max_nodes:
                for x in X:
                    self.visited.append(c)
                    c+=1
                    self.train_single(x)
                    sys.stdout.write('\r epoch %i / %i :  %i%% : nodes - %i' %(i+1, iterations,c*100/t, len(self.neurons) ))
                    sys.stdout.flush()
            else:
                self.batch_nn_train(X, self.lr)
                sys.stdout.write('\r epoch %i / %i : nodes - %i' %((i+1), iterations, len(self.neurons)))
                sys.stdout.flush()
            self.lr *= (1 - 3.8 / len(self.neurons))
            self.range = (self.radius -1)  * np.exp(- i / iterations) + 1.00001
        if prune:
            for k in self.neurons.keys():
                if len(self.neurons) > self.min_nodes and ( self.hits[k] < X.shape[0]/10.0 or self.gen[k] < self.current_gen * 0.75 ) :
                    del self.gen[k]
                    del self.errors[k]
                    del self.neurons[k]
                    del self.grid[k]
                    del self.hits[k]

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
        hs = scale(hs, with_mean=False)
        hs /= hs.max()
        # hs.fill(1)
        weights = np.array(self.neurons.values())[neighbors]

        weights += np.array([hs]).T * (x - weights) * self.lr

        for neighbor, w in zip(np.array(self.neurons.keys())[neighbors], weights):
            self.neurons[neighbor] = w #* self.lr * (x - self.w[neighbor])
            # self.w[neighbor] +=  self.lr* (x - self.w[neighbor]) #* self.lr * (x - self.w[neighbor])
            # try:
            #     if self.errors[neighbor] > self.GT and self.max_nodes > len(self.w):
            #         self.grow(neighbor)
            # except:
            #     continue
        try:
            self.errors[bmu] += err
        except KeyError:
            self.errors[bmu] = err
        if self.errors[bmu] > self.Herr:
                self.Herr += err

        for k in np.array(self.neurons.keys())[neighbors]:
            if k == bmu and self.errors[k] > self.GT and self.max_nodes > len(self.neurons):
                self.current_gen += 1
                self.grow(k)

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

    def find_closest_visited(self, x):

        try:
            visited = np.array(self.visited)
            X_ = self.X[visited]
            bmv = visited[np.linalg.norm(X_ - x, axis=1).argmin()]
            return self.X[bmv]
        except:
            return x

    def find_knn_mean(self, x, k = 2):

        nns = np.linalg.norm(x - self.X).argsort()[:k+1]
        return self.X[nns].mean(axis=0)

    def find_bmu(self, x):
        nodes = np.asarray(self.neurons.values())
        deltas = nodes - self.find_closest_visited(x)
        dist_sqr = np.sum(deltas**2, axis =1 )
        mink = np.argmin(dist_sqr)
        # mink = pairwise_distances_argmin(nodes, np.array([x]))
        try:
            dist =minkowski(self.neurons.values()[mink], x, p = 2)
        except ValueError:
            print 'nan'

        self.hits[self.neurons.keys()[mink]] += 1
        return self.neurons.keys()[mink], dist  #dist_sqr[mink]

    def get_neighbourhood(self, node):
        p_dist_matrix = pairwise_distances(np.array(self.grid.values()))
        #np.fill_diagonal(p_dist_matrix, np.Infinity)
        node_dists = p_dist_matrix[np.where(np.array(self.neurons.keys())==node)[0]][0]
        return np.where(node_dists< self.range)[0], node_dists[np.where(node_dists<self.range)[0]]#np.array(self.grid.keys())


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
                self.errors[str(list(nei))] += self.errors[bmu] * self.fd
            except KeyError:
                w = self.get_new_weight(bmu, nei)

                self.neurons[str(list(nei))] = w
                self.grid[str(list(nei))] = list(nei)
                self.errors[str(list(nei))] = self.GT/2
                self.gen[str(list(nei))] = self.current_gen
                self.hits[str(list(nei))] = 0
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
                w2 = self.neurons[self.grid.keys()[second_neighbours]]
            except:
                try:
                    w2 = self.neurons[self.grid.keys()[third_neighbours[0]]]
                except:
                    w2 = np.random.random(self.dims) * 2 -1
            return 2 * w1 - w2
        # elif order2.shape[0] > 0:
        #     try:
        #         w2 = self.w[self.grid.keys()[np.where(np.linalg.norm(np.array(self.grid.values())[order2] - np.array(self.grid[old]), axis=1)==1)[0]]]
        #     except :
        #         w2 = np.random.random(self.dims)
        #     return 2*w1 - w2




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
            anc = {0: np.array([0, 1]),
                   1: np.array([1, 0]),
                   2: np.array([0, -1]),
                   3: np.array([-1, 0])}
            w1 = self.neurons[str(list((nei - anc[(direction)])))]
            w2 = self.neurons[str(list(nei - 2 * anc[(direction)]))]
            return 2 * w1 - w2
        except KeyError:
            return np.array([0])


    def type_c(self, nei, direction):
        try:
            anc = {0: np.array([0, 1]),
                   1: np.array([1, 0]),
                   2: np.array([0, -1]),
                   3: np.array([-1, 0])}

            if direction == 0 or direction == 2:
                try:
                    return 2 * self.neurons[str(list(nei - anc[direction]))] - self.neurons[
                        str(list(nei - anc[direction] + np.array([1, 0])))]
                except KeyError:
                    return 2 * self.neurons[str(list(nei - anc[direction]))] - self.neurons[
                        str(list(nei - anc[direction] + np.array([-1, 0])))]

            else:
                try:
                    return 2 * self.neurons[str(list(nei - anc[direction]))] - self.neurons[
                        str(list(nei - anc[direction] + np.array([0, 1])))]
                except KeyError:
                    return 2 * self.neurons[str(list(nei - anc[direction]))] - self.neurons[
                        str(list(nei - anc[direction] + np.array([0, -1])))]
        except KeyError:
            return np.array([0])

