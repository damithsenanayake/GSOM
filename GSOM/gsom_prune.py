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
        self.cache = []
        self.gen = {}
        self.means={}
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

    def add_to_cache(self, k):
        if k in self.cache:
            del self.cache[self.cache.index(k)]
        self.cache.append(k)

    def remove_node(self, k):
        del self.cache[self.cache.index(k)]
        del self.neurons[k]
        del self.grid[k]
        del self.errors[k]
        del self.gen[k]
        # del self.hits[k]

    def prune_LRU(self):
        for k in self.cache[:int(len(self.cache)*0.01)]:
            self.remove_node(k)

    def train_batch(self, X, iterations, lr, prune = True):
        if not lr == 0:
            self.lr = lr
        for i in range(iterations):
            c = 0
            t = X.shape[0]
            for x in X:
                c+=1
                self.train_single(x)
                sys.stdout.write('\r epoch %i / %i :  %i%% : nodes - %i' %(i+1, iterations,c*100/t, len(self.neurons) ))
                sys.stdout.flush()

            for k in np.array(self.neurons.keys()):

                if self.errors[k] > self.GT:
                    self.grow(k)

            self.lr *= (1 - 3.8 / len(self.neurons))
            self.range = (self.radius -1)  * np.exp(- i / iterations) + 1.00001

        self.prune_LRU()



    def train_single(self, x):
        bmu, err = self.find_bmu(x)

        self.add_to_cache(bmu)

        neighbors , dists = self.get_neighbourhood(bmu)
        hs = np.exp(-(dists / 2*self.range))
        hs = scale(hs, with_mean=False)
        hs /= hs.max()
        # hs.fill(1)
        weights = np.array(self.neurons.values())[neighbors]

        weights += np.array([hs]).T * (x - weights) * self.lr

        for neighbor, w in zip(np.array(self.neurons.keys())[neighbors], weights):
            self.neurons[neighbor] = w
        try:
            self.errors[bmu] += err
        except KeyError:
            self.errors[bmu] = err
        if self.errors[bmu] > self.Herr:
                self.Herr += err


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


