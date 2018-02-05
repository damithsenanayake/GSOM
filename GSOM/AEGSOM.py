import numpy as np
from scipy.spatial.distance import minkowski
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale
import sys
from AutoEncoder import AutoEncoder
def sig(x):
    return 1/(1+np.exp(-x))

class GSOM(object):

    def __init__(self, dims, hid,  sf, fd, max_nodes, min_nodes, radius = 5, scale=1,X=None, nei=True, gaussian=False, map_init = 2, lr = 1):
        self.lr = lr
        self.nei = nei

        self.s1 = None,
        self.m1 = None
        if X != None:
            self.s1 = X.std()
            self.m1 = X.mean()

        self.radius = radius
        self.gaussian = gaussian
        self.scale = scale
        self.hid = hid
        self.w1 = {}
        self.w2 = {}
        self.b1 = {}
        self.b2 = {}
        self.grid = {}
        self.errors = {}
        self.hits = {}
        self.gen = {}
        self.learners = {}
        for i in range(map_init):
            for j in range(map_init):
                AE = AutoEncoder(dims, hid, self.s1, self.m1,  gaussian)
                self.learners[str([i, j])] = AE
                self.grid[str([i, j])] = [i, j]
                self.errors[str([i, j])] = 0
                self.gen[str([i, j])] = 0
                self.hits[str([i, j])] = 0


        self.sf = sf
        self.GT = -np.log(sf)*dims# * /255.0
        self.fd = fd
        self. max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.dims = dims
        self.range = radius
        self.Herr = 0
        self.current_gen = 0
        self.learn_iters = 0

    def param_learn_smooth(self, x, node, h, lr):
        self.learners.values()[node].train(np.array([x]), 50, h * lr, momentum=0.25,
                                           wd_param=0.01)  # min(1,self.learn_iters),h*lr, momentum=0.5, wd_param = 0.1)
    def param_learn(self, x, node, h, lr):
        self.learners.values()[node].train( np.array([x]),1 ,h*lr, momentum=0.75, wd_param = 0.75)#min(1,self.learn_iters),h*lr, momentum=0.5, wd_param = 0.1)

    def reconstruct(self, x, node):
        return self.learners[node].predict(np.array([x]))*self.scale

    def smooth_batch(self, X, iterations, lr, prune=True):
        if lr:
            self.lr = lr
        self.learn_iters += 1
        self.same_count = 0
        self.previous = None
        for i in range(iterations):
            self.lr = lr
            c = 0
            t = X.shape[0]
            for x in X:
                c += 1
                self.train_single_smooth(x)
                sys.stdout.write(
                    '\r epoch %i / %i :  %i%% : nodes - %i' % (i + 1, iterations, c * 100 / t, len(self.grid)))
                sys.stdout.flush()
            #
            #     self.lr *= (1 - 3.8 / len(self.learners))
            # self.range = (self.radius - 1) * np.exp(- i * 2 / iterations) + 1.00001

    def prune(self, k = None):
        if k != None:
            del self.gen[k]
            del self.errors[k]
            del self.learners[k]
            del self.grid[k]
            del self.hits[k]
            return
        for k in self.grid.keys():
            if  (self.hits[k] < 1 or self.gen[k] < self.current_gen * 0.8):
                del self.gen[k]
                del self.errors[k]
                del self.learners[k]
                del self.grid[k]
                del self.hits[k]

    def train_batch(self, X, iterations, lr, prune):
        # for k in self.grid.keys():
        #     self.hits[k] = 0
        if lr:
            self.lr = lr
        self.learn_iters += 1
        self.same_count = 0
        self.previous = None
        for i in range(iterations):
            self.lr = lr

            c = 0
            lamda = (i+1) / iterations
            t = X.shape[0]
            for x in X:
                c+=1
                self.train_single(x)
                sys.stdout.write('\r epoch %i / %i :  %i%% : nodes - %i' %(i+1, iterations,c*100/t, len(self.grid) ))
                sys.stdout.flush()

                self.lr *= (1 - 3.8 / len(self.learners))
            self.range = (self.radius -1)  * np.exp(- i * 2 / iterations) + 1.00001
            # self.GT *= 0.1


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

    def train_single_smooth(self, x):
        bmu, err = self.find_bmu(x, False)

        neighbors, dists = self.get_neighbourhood_smooth(bmu)
        # hs = np.exp(-(dists**2 / (2*self.range**2)))#/np.sum(np.exp(-(dists**2 / (2*self.range**2))))
        hs = np.exp(-(dists / 2 * self.range))
        hs = scale(hs, with_mean=False)
        hs /= hs.max()

        for neighbor, h in zip(neighbors, hs):
            self.param_learn_smooth(x, neighbor, h, self.lr)



    def train_single(self, x):
        bmu, err = self.find_bmu(x, False)

        neighbors , dists = self.get_neighbourhood(bmu)
        # hs = np.exp(-(dists**2 / (2*self.range**2)))#/np.sum(np.exp(-(dists**2 / (2*self.range**2))))
        hs = np.exp(-(dists / 2 * self.range))
        hs = scale(hs, with_mean=False)
        hs /= hs.max()

        for neighbor, h in zip(neighbors, hs):
            #if self.hits.values()[neighbor] < 50: #### This gave good results. See if this had any effects!
            if not self.nei:
                h = 1

            self.param_learn(x, neighbor, h, self.lr)

        try:
            self.errors[bmu] += err
        except KeyError:
            self.errors[bmu] = err
        if self.errors[bmu] > self.Herr:
                self.Herr =self.errors[bmu]
        if self.Herr > self.GT and self.max_nodes > len(self.grid):
            self.Herr = self.GT
            for k in np.array(self.grid.keys())[neighbors]:
                if k == bmu and self.errors[k] > self.GT and self.max_nodes > len(self.grid):
                    self.current_gen += 1
                    self.grow(k)

    def predict(self, X):
        arr = []
        hits = []
        for x in X:
            hit , dist = self.find_bmu(x, reg=False)
            m = []
            for n in np.array(self.learners.keys())[self.get_neighbours(hit)[0]]:
                if n != hit:
                    m.append(np.exp(-np.linalg.norm(x-self.reconstruct(x, n)))*(np.array(self.grid[hit])-np.array(self.grid[n])))
            b = self.grid[hit]
            b += np.array(m).sum(axis=0)
            hits.append(hit)
            try:
                self.hits[hit] += 1
            except KeyError:
                self.hits[hit] = 1
            arr.append(b)

        return np.array(arr), hits


    def find_bmu(self, x, reg=True):
        dropout_node = -1
        if reg:
            ps = np.array(self.hits.values()).astype(float)
            if not ps.any():
                ps += 1.0/(ps.shape[0])
            ps /= ps.sum()
            #dropout regularization for the map
            dropout_node = np.random.choice(ps.shape[0], 1, p = ps)

        recs = []
        for k in self.learners.keys():
            recs.append(np.array(self.reconstruct(x,k))[0])

        diffs = (np.array(np.asarray(recs)) -x)**2

        dists = np.linalg.norm(diffs, axis = 1)
        bmu = dists.argmin()
        if bmu == dropout_node:
            bmu = dists.argsort()[1]


        try:
            self.hits[self.learners.keys()[bmu]] += 1
        except KeyError:
            self.hits[self.learners.keys()[bmu]] = 1
        return self.learners.keys()[bmu], np.sqrt(dists[bmu])

    def get_neighbourhood(self, node):
        p_dist_matrix = pairwise_distances(np.array(self.grid.values()))
        #np.fill_diagonal(p_dist_matrix, np.Infinity)
        node_dists = p_dist_matrix[np.where(np.array(self.grid.keys())==node)[0]][0]
        return np.where(node_dists< self.range)[0], node_dists[np.where(node_dists<self.range)[0]]#np.array(self.grid.keys())

    def get_neighbourhood_smooth(self, node):
        p_dist_matrix = pairwise_distances(np.array(self.grid.values()))
        # np.fill_diagonal(p_dist_matrix, np.Infinity)
        node_dists = p_dist_matrix[np.where(np.array(self.grid.keys()) == node)[0]][0]
        return np.where(node_dists < 1.1)[0], node_dists[np.where(node_dists < 1.1)[0]]

    def get_neighbours(self, node):
        p_dist_matrix = pairwise_distances(np.array(self.grid.values()))
        # np.fill_diagonal(p_dist_matrix, np.Infinity)
        node_dists = p_dist_matrix[np.where(np.array(self.grid.keys()) == node)[0]][0]
        return np.where(node_dists < 1.1)[0], node_dists[
            np.where(node_dists < 1.1)[0]]  # np.array(self.grid.keys())

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
                w1, w2, b1, b2 = self.get_new_weight(bmu, nei)
                AE = AutoEncoder(self.dims, self.hid, self.s1, self.m1, gaussian=self.gaussian)

                AE.set_params(w1, w2, b1, b2)

                self.learners[str(list(nei))] = AE
                self.grid[str(list(nei))] = list(nei)
                self.errors[str(list(nei))] = self.GT/2
                self.gen[str(list(nei))] = self.current_gen
                self.hits[str(list(nei))] = 0
        self.errors[bmu] = self.GT / 2

    def get_new_weight(self, old, new):

        grid = np.array(self.grid.values())
        dists = np.linalg.norm(grid - np.array(new), axis=1)
        order1 = np.where(dists == 1)[0]
        order2 = np.where(dists == 2)[0]
        order2L = np.where(dists == np.sqrt(2))[0]
        n1 = self.learners[old]
        # if order1.shape[0] > 1:
        try:
            n2 = self.learners[self.grid.keys()[order1[
                np.where(np.linalg.norm(np.array(self.grid.values())[order1] - np.array(self.grid[old]), axis=1) == 2)[
                    0]]]]
            return (n1.w1 + n2.w1) / 2, (n1.w2+n2.w2) / 2, (n1.b1 + n2. b1) / 2, (n1.b2 + n2. b2)/2
        except TypeError:
            second_neighbours = order2[
                np.where(np.linalg.norm(np.array(self.grid.values())[order2] - np.array(self.grid[old]), axis=1) == 1)[
                    0]]
            third_neighbours = order2L[
                np.where(np.linalg.norm(np.array(self.grid.values())[order2L] - np.array(self.grid[old]), axis=1) == 1)[
                    0]]
            try:
                n2 = self.learners[self.grid.keys()[second_neighbours]]
            except:
                try:
                    n2 = self.learners[self.grid.keys()[third_neighbours[0]]]
                except:
                    n2 = AutoEncoder(self.dims, self.hid, self.s1, self.m1, gaussian=self.gaussian)
            return 2 * n1.w1 - n2.w1, 2* n1.w2 - n2.w2, 2 * n1.b1 - n2.b1, 2*n1.b2 - n2.b2


    def type_b(self, nei, direction):
        try:
            if direction == 0 or direction == 2:

                return (self.learners[str(list(nei + np.array([0, -1])))].w1+ self.learners[
                    str(list((nei + np.array([0, 1]))))].w1) * 0.5, (self.learners[str(list(nei + np.array([0, -1])))].w2+ self.learners[
                    str(list((nei + np.array([0, 1]))))].w2) * 0.5, (self.learners[str(list(nei + np.array([0, -1])))].b1+ self.learners[
                    str(list((nei + np.array([0, 1]))))].b1) * 0.5, (self.learners[str(list(nei + np.array([0, -1])))].b2+ self.learners[
                    str(list((nei + np.array([0, 1]))))].b2) * 0.5

            return (self.learners[str(list(nei + np.array([-1, 0])))].w1 + self.learners[
                str(list(nei + np.array([1, 0])))].w1) * 0.5, (self.learners[str(list(nei + np.array([-1, 0])))].w2 + self.learners[
                str(list(nei + np.array([1, 0])))].w2) * 0.5, (self.learners[str(list(nei + np.array([-1, 0])))].b1 + self.learners[
                str(list(nei + np.array([1, 0])))].b1) * 0.5, (self.learners[str(list(nei + np.array([-1, 0])))].b2 + self.learners[
                str(list(nei + np.array([1, 0])))].b2) * 0.5
        except KeyError:
            return np.array([0])


    def type_a(self, nei, direction):
        try:
            anc = {0: np.array([0, 1]),
                   1: np.array([1, 0]),
                   2: np.array([0, -1]),
                   3: np.array([-1, 0])}
            wa1 = self.learners[str(list((nei - anc[(direction)])))].w1
            wa2 = self.learners[str(list(nei - 2 * anc[(direction)]))].w1
            wb1 = self.learners[str(list((nei - anc[(direction)])))].w2
            wb2 = self.learners[str(list(nei - 2 * anc[(direction)]))].w2
            ba1 = self.learners[str(list((nei - anc[(direction)])))].b1
            ba2 = self.learners[str(list(nei - 2 * anc[(direction)]))].b1
            bb1 = self.learners[str(list((nei - anc[(direction)])))].b2
            bb2 = self.learners[str(list(nei - 2 * anc[(direction)]))].b2

            return 2 * wa1 - wa2, 2*wb1 - wb2, 2 * ba1 - ba2, 2* bb1 - bb2
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
                    w1 =2 * self.learners[str(list(nei - anc[direction]))].w1 - self.learners[
                        str(list(nei - anc[direction] + np.array([1, 0])))].w1
                    w2 =2 * self.learners[str(list(nei - anc[direction]))].w2 - self.learners[
                        str(list(nei - anc[direction] + np.array([1, 0])))].w2
                    b1 =2 * self.learners[str(list(nei - anc[direction]))].b1 - self.learners[
                        str(list(nei - anc[direction] + np.array([1, 0])))].b1
                    b2 =2 * self.learners[str(list(nei - anc[direction]))].b2 - self.learners[
                        str(list(nei - anc[direction] + np.array([1, 0])))].b2

                    return w1,w2 , b1 , b2
                except KeyError:

                    return 2 * self.learners[str(list(nei - anc[direction]))].w1 - self.learners[
                        str(list(nei - anc[direction] + np.array([-1, 0])))].w1,2 * self.learners[str(list(nei - anc[direction]))].w2 - self.learners[
                        str(list(nei - anc[direction] + np.array([-1, 0])))].w2,2 * self.learners[str(list(nei - anc[direction]))].b1 - self.learners[
                        str(list(nei - anc[direction] + np.array([-1, 0])))].b1,2 * self.learners[str(list(nei - anc[direction]))].b2 - self.learners[
                        str(list(nei - anc[direction] + np.array([-1, 0])))].b2

            else:
                try:

                    if np.isnan(2 * self.learners[str(list(nei - anc[direction]))].w1 - self.learners[
                        str(list(nei - anc[direction] + np.array([0, 1])))].w1).any():
                        print 'shit'

                    return 2 * self.learners[str(list(nei - anc[direction]))].w1 - self.learners[
                        str(list(nei - anc[direction] + np.array([0, 1])))].w1, 2 * self.learners[str(list(nei - anc[direction]))].w2 - self.learners[
                        str(list(nei - anc[direction] + np.array([0, 1])))].w2, 2 * self.learners[str(list(nei - anc[direction]))].b1 - self.learners[
                        str(list(nei - anc[direction] + np.array([0, 1])))].b1, 2 * self.learners[str(list(nei - anc[direction]))].b2 - self.learners[
                        str(list(nei - anc[direction] + np.array([0, 1])))].b2
                except KeyError:
                    if np.isnan(2 * self.learners[str(list(nei - anc[direction]))].w1 - self.learners[
                        str(list(nei - anc[direction] + np.array([0, -1])))].w1).any():
                        print 'shit'
                    return 2 * self.learners[str(list(nei - anc[direction]))].w1 - self.learners[
                        str(list(nei - anc[direction] + np.array([0, -1])))].w1, 2 * self.learners[str(list(nei - anc[direction]))].w2 - self.learners[
                        str(list(nei - anc[direction] + np.array([0, -1])))].w2, 2 * self.learners[str(list(nei - anc[direction]))].b1 - self.learners[
                        str(list(nei - anc[direction] + np.array([0, -1])))].b1, 2 * self.learners[str(list(nei - anc[direction]))].b2 - self.learners[
                        str(list(nei - anc[direction] + np.array([0, -1])))].b2
        except KeyError:
            return np.array([0])









