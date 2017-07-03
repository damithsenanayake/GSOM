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

    def __init__(self, dims, hid,  sf, fd, max_nodes, min_nodes, radius = 5, scale=1,X=None):

        self.s1 = None,
        self.m1 = None
        if X != None:
            self.s1 = X.std()
            self.m1 = X.mean()

        self.radius = radius

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
        for i in range(2):
            for j in range(2):
                AE = AutoEncoder(dims, hid, self.s1, self.m1)
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

    def param_learn(self, x, node, h, lr):
        self.learners.values()[node].train( np.array([x]), min(2,self.learn_iters),h*lr, momentum=0.5, wd_param = 0.1)

    def reconstruct(self, x, node):
        return self.learners[node].predict(np.array([x]))*self.scale



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
        self.lr = lr
        self.learn_iters += 1
        self.same_count = 0
        self.previous = None
        for i in range(iterations):
            c = 0
            lamda = (i+1) / iterations
            t = X.shape[0]
            if True:#len(self.w) < self.max_nodes:
                for x in X:
                    c+=1
                    self.train_single(x)
                    sys.stdout.write('\r epoch %i / %i :  %i%% : nodes - %i' %(i+1, iterations,c*100/t, len(self.grid) ))
                    sys.stdout.flush()
            else:
                self.batch_nn_train(X, self.lr)
                sys.stdout.write('\r epoch %i / %i : nodes - %i' %((i+1), iterations, len(self.grid)))
                sys.stdout.flush()
            self.lr *= (1 - 3.8 / len(self.grid))
            if self.range > 1.414:
                self.range = self.radius * np.exp(- min(self.learn_iters,10) / 10.0)
            if prune:
                hits = np.array(self.hits.values())
                mean = hits.mean()
                std = hits.std()
                upper = mean + 3 * std
                lower = mean - 3 *std

                for k in self.grid.keys():
                    if  len(self.hits.values()) > self.min_nodes and ( self.hits[k] < max(lower, 1) or self.hits[k] > upper or self.gen[k] < self.current_gen * 0.5  * np.exp(-i / iterations)) :#and len(self.grid) > or self.gen[k] < self.current_gen * 0.75 self.min_nodes   :
                        del self.gen[k]
                        del self.errors[k]
                        del self.learners[k]
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
        hs = np.exp(-(dists / (2*self.range)))/np.sum(np.exp(-(dists / (2*self.range))))
        ts = np.random.binomial(1, np.exp(-(dists**2 / (2*self.range**2))), size=hs.shape)

        for neighbor, t, h in zip(neighbors, ts, hs):
            if t:
                if self.hits.values()[neighbor] < 50:
                    self.param_learn(x, neighbor, h, self.lr)

        try:
            self.errors[bmu] += err
        except KeyError:
            self.errors[bmu] = err
        if self.errors[bmu] > self.Herr:
                self.Herr += err
        if self.errors[bmu] > self.GT and self.max_nodes > len(self.grid):
            self.current_gen += 1
            self.grow(bmu)
        for k in np.array(self.grid.keys())[neighbors]:
            if self.errors[k] > self.GT and self.max_nodes > len(self.grid):
                self.current_gen += 1
                self.grow(k)

    def predict(self, X):
        arr = []
        hits = []
        for x in X:
            hit = self.find_bmu(x, reg=False)[0]
            hits.append(hit)
            try:
                self.hits[hit] += 1
            except KeyError:
                self.hits[hit] = 1
            arr.append(self.grid[hit])

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


    def grow(self, bmu):
        # type: (object) -> object
        p = self.grid[bmu]
        up = p + np.array([0, +1])
        right = p + np.array([+1, 0])
        down = p + np.array([0, -1])
        left = p + np.array([-1, 0])

        neighbors = np.array([up, right, down, left])
        direction = 0
        for nei in neighbors:
            try:
                self.errors[str(list(nei))] += self.errors[bmu] * self.fd
            except KeyError:
                try:
                    w1, w2, b1, b2 = self.type_b(nei, direction)
                    if np.isnan(w1).any():
                        print 'shit'
                except:
                    try:
                        w1, w2, b1, b2 = self.type_a(nei, direction)
                        if np.isnan(w1).any():
                            print 'shit'
                    except:
                        try:
                            w1, w2, b1, b2 = self.type_c(nei, direction)
                            if np.isnan(w1).any():
                                print 'shit'

                        except:
                            w1 = None
                            w2 = None#np.ones((self.hid, self.dims))
                            b1 = None#np.ones(self.hid)
                            b2 = None#np.ones(self.dims)

                #     if new_a.any():
                #         if newf_c.any():
                #             # w.fill(0.5)
                #         else:
                #             w = new_c
                #     else:
                #         w = new_a
                # else:
                #     w = new_b

                AE = AutoEncoder(self.dims, self.hid, self.s1, self.m1)
                # if np.isnan(w1).any():
                #     print 'shit'
                AE.set_params(w1, w2, b1, b2)

                self.learners[str(list(nei))] = AE
                self.grid[str(list(nei))] = list(nei)
                self.errors[str(list(nei))] = self.GT/2
                self.gen[str(list(nei))] = self.current_gen
                self.hits[str(list(nei))] = 0
            direction += 1
        self.errors[bmu] = self.GT / 2

    def type_b(self, nei, direction):
        try:
            if direction == 0 or direction == 2:
                if np.isnan((self.learners[str(list(nei + np.array([0, -1])))].w1+ self.learners[
                    str(list((nei + np.array([0, 1]))))].w1)).any():
                    print "shit1"

                return (self.learners[str(list(nei + np.array([0, -1])))].w1+ self.learners[
                    str(list((nei + np.array([0, 1]))))].w1) * 0.5, (self.learners[str(list(nei + np.array([0, -1])))].w2+ self.learners[
                    str(list((nei + np.array([0, 1]))))].w2) * 0.5, (self.learners[str(list(nei + np.array([0, -1])))].b1+ self.learners[
                    str(list((nei + np.array([0, 1]))))].b1) * 0.5, (self.learners[str(list(nei + np.array([0, -1])))].b2+ self.learners[
                    str(list((nei + np.array([0, 1]))))].b2) * 0.5
            if np.isnan((self.learners[str(list(nei + np.array([-1, 0])))].w1 + self.learners[
                str(list(nei + np.array([1, 0])))].w1) * 0.5).any():
                print 'shit2'
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

            if np.isnan(2 * wa1 - wa2).any():
                print 'shit'

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

                    if np.isnan(w1).any():
                        print 'shit'


                    return w1,w2 , b1 , b2
                except KeyError:
                    if np.isnan(2 * self.learners[str(list(nei - anc[direction]))].w1 - self.learners[
                        str(list(nei - anc[direction] + np.array([-1, 0])))].w1).any():
                        print 'shit'

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


