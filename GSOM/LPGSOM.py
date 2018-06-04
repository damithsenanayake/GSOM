import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin
import timeit

import sys

class GSOM(object):

    def __init__(self,lr = .1,  beta=0.45, sf=0.9, wd=0.04, fd=1.9):
        self.radius = 0
        self.lr = 0
        self.neurons = {}
        self.grid = {}
        self.errors = {}
        self.hits = {}
        self.gen = {}
        self.cache = []
        self.sf = sf
        self.beta = beta
        self.ages = {}
        self.fd = fd
        self.amax = 3
        self.lr = lr
        self.means={}

        self.wd = wd

        self.p_dist_matrix = None
        self.grid_changed = True
        self.GT = 0

        self.dims = 0
        self.range = 0
        self.Herr = 0
        self.current_gen = 0

    def fit(self, X ):
        self.start_time = timeit.default_timer()
        self.dims = X.shape[1]
        self.GT = -self.dims * np.log(self.sf)  # /255.0
        init_vect = np.random.random(self.dims)
        self.radius = 10# np.exp(1)
        for i in range(70):
            for j in range(70):
                self.neurons[str([i, j])] = np.random.random(self.dims)
                self.grid[str([i, j])] = [i, j]
                self.errors[str([i, j])] = 0


        st = timeit.default_timer()
        # self.train_batch(X[np.random.permutation(np.array(range(X.shape[0])))])
        self.lr/=2.
        et = timeit.default_timer() - st
        print "\n elapsed time for growing : ", et , "\n"
        self.Y = np.array(self.grid.values()).astype(float)
        self.Y -= self.Y.min(axis=0)
        self.Y /= self.Y.max(axis=0)
        self.C = np.array(self.neurons.values())
        self.n_graph = np.zeros((self.Y.shape[0], self.Y.shape[0]))

        gridar = np.array(self.grid.values())

        for n in range(self.n_graph.shape[0]):
            self.n_graph[np.where(np.linalg.norm(gridar[n]-gridar, axis=1)==1)[0]]=1

        self.smoothen_wd(X)





    def predict_inner(self, X):
        hits = []
        for x in X:
            hit = np.argmin(np.linalg.norm(x - self.C, axis=1))
            hits.append(hit)

        return np.array(hits)

    def fit_transform(self, X ):
        self.fit(X)
        return self.LMDS(X)

    def train_batch(self, X):
        i = 0
        lr = self.lr
        self.lrst  = lr
        self.spawns = 0
        rad = self.radius
        wd_orig = self.wd
        its = 10
        self.it_rate = 0
        while i< its:#self.lr > 0.5*lr:
            c = 0
            t = X.shape[0]
            self.Herr=0
            self.hits = {}
            self.it_rate = (i+1.)/its
            if i:
                Xtr = X[np.random.choice(range(t),np.floor(1./its *(1+i)*t).astype(int)).astype(int)]
            else:
                Xtr = X
            self.wd = wd_orig * ((i+1)*1./its)
            for x in Xtr:
                c+=1
                self.train_single(x)
                while self.Herr >= self.GT and not(i) :  # and i<=5:
                    growinds = np.where(np.array(self.errors.values()) >= self.GT)[0]
                    self.grid_changed = 1
                    for g in growinds:
                        self.grow(self.errors.keys()[g])
                    self.Herr = np.array(self.errors.values()).max()
                et = timeit.default_timer() - self.start_time

                sys.stdout.write('\r epoch %i :  %i%% : nodes - %i : LR - %s : radius : %s : time : %s' %(i+1, c*100/t, len(self.neurons), str(self.lr), str(self.radius), str(et)))
                sys.stdout.flush()
                self.Herr = np.array(self.errors.values()).max()

            if self.radius <=1:
                break
            self.Herr = np.array(self.errors.values()).max()

            while self.Herr >= self.GT and i%3:#and i<=5:
                growinds = np.where(np.array(self.errors.values())>=self.GT)[0]
                self.grid_changed=1
                for g in growinds:
                    self.grow(self.errors.keys()[g])
                self.Herr = np.array(self.errors.values()).max()
            self.lr *= 0.9 #* (1 - 3.8 / len(self.neurons))  # np.exp(-i/50.0)#
            # self.radius = rad*np.exp(-2.*i/float(40))  # (1 - 3.8 / len(self.w))
            # self.radius = rad *(1-i*1./its)
            for k in self.errors.keys():
                self.errors[k] = 0
            i += 1

            # self.prune()
        self.wd = wd_orig
        print 'spawns ',self.spawns



    def train_single(self, x):
        # bmu, err = self.find_bmu(x)
        W = np.array(self.neurons.values())
        winner = pairwise_distances_argmin(np.array([x]), W, axis=1)[0]
        bmu = self.neurons.keys()[winner]
        # neighbors , dists = self.get_neighbourhood(bmu)
        # self.fd = 1.0 / len(neighbors)
        l_dists = np.linalg.norm(np.array(self.grid.values())-np.array(self.grid.values()[winner]), axis=1)
        neighbors = np.where(l_dists<self.radius)[0]#np.argsort(l_dists)[:20]
        dists = l_dists[neighbors]
        h_dists = np.linalg.norm(np.array(self.neurons.values())-np.array(self.neurons[bmu]), axis=1)[neighbors]
        theta_D = np.array([1-np.exp(-15.5*h_dists**40 / (h_dists.max())**40)]).T
        hs = np.array([np.exp(-10.5*dists**2/(self.radius**2))]).T
        # hs.fill(1)
        weights = np.array(self.neurons.values())[neighbors]

        weights += (x - weights) * self.lr*hs
        weights -= weights * theta_D * self.wd# - theta_D*self.wd*weights#*self.it_rate#- (1-1*self.lr/self.lrst)*self.lr * self.wd*weights
        err = np.linalg.norm(W[winner]-x)
        for neighbor, w in zip(np.array(self.neurons.keys())[neighbors], weights):
            self.neurons[neighbor] = w
        try:
            self.errors[bmu] += err
        except KeyError:
            self.errors[bmu] = err

    def predict(self, X):
        hits =[]
        for x in X:
            hit = self.Y[np.argmin(np.linalg.norm(x-self.C, axis = 1))]
            hits.append(hit)

        return np.array(hits)

    def prune(self):
        # keys = np.array(self.hits.keys())
        # hits = np.array(self.hits.values())
        # hitmean = np.median(hits)
        # hitstde = np.std(hits)
        #
        # del_cands = keys[np.where(hits < (hitmean + 0.5* hitstde))[0]]
        age_vals = np.array(self.ages.values())
        keys = np.array(self.grid.keys())

        del_cands = np.where(age_vals >= self.amax)[0]
        i = 0
        for ind in np.array(self.grid.keys())[del_cands]:
            del self.neurons[ind]
            del self.grid[ind]
            del self.errors[ind]
            del self.ages[ind]
            i+=1
        self.grid_changed =1# (del_cands.shape[0])
        sys.stdout.write('\ndeleted nodes : %i' % i)






    def find_bmu(self, x):
        nodes = np.asarray(self.neurons.values())
        mink= pairwise_distances_argmin(np.array([x]), nodes, axis=1)[0]# = np.linalg.norm(x - nodes, axis=1)
        # mink = np.argmin(dists)
        dist =np.linalg.norm(x - nodes, axis=1)[mink]
        # mink = pairwise_distances_argmin(nodes, np.array([x]))
        # try:
        #     dist =minkowski(self.neurons.values()[mink], x, p = 2)
        # except ValueError:
        #     print 'nan'

        self.ages[self.neurons.keys()[mink]] = 0

        try:
            self.hits[mink] += 1
        except KeyError:
            self.hits[mink] = 1
        return self.neurons.keys()[mink], dist   #dist_sqr[mink]

    def get_neighbourhood(self, node):
        if self.grid_changed:
            self.p_dist_matrix = pairwise_distances(np.array(self.grid.values()))
            self.grid_changed=False
        node_dists = self.p_dist_matrix[np.where(np.array(self.neurons.keys()) == node)[0]][0]
        immNeis = np.where(node_dists<=2.0)[0]
        neis = np.where(node_dists < self.radius)[0]

        for n in neis:
            try :
                self.ages[self.neurons.keys()[n]] += 1
            except :
                self.ages[self.neurons.keys()[n]] = 1
        for im in immNeis:
            self.ages[self.neurons.keys()[im]] = 0

        return np.where(node_dists < self.radius)[0], node_dists[
            np.where(node_dists < self.radius)[0]]  # np.array(self.grid.keys())



    def grow(self, bmu):
        # type: (object) -> object
        # print ' n(G) = ',len(self.neurons.keys()),
        p = self.grid[bmu]
        up = p + np.array([0, +1])
        right = p + np.array([+1, 0])
        down = p + np.array([0, -1])
        left = p + np.array([-1, 0])

        neighbors = np.array([up, right, down, left])
        for nei in neighbors:
            try:
                self.errors[str(list(nei))] += self.errors[str(list(nei))] * self.fd
            except KeyError:
                self.grid_changed=True
                w = self.get_new_weight(bmu, nei)

                self.neurons[str(list(nei))] = w
                self.grid[str(list(nei))] = list(nei)
                self.errors[str(list(nei))] = self.errors[bmu]*self.fd
                self.ages[str(list(nei))]=0

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
                    w2 = w1*0.9# np.random.random(self.dims) * 2 -1
                    self.spawns +=1
            return 2 * w1 - w2

    def smoothen_wd(self, X):
        self.thet_vis_bundle = {}
        r_st = .5
        its = 6000

        lr = self.lr
        print self.wd
        st = timeit.default_timer()
        grid_dists = pairwise_distances(self.Y, self.Y)
        self.cand_hits = np.ones(shape=(self.Y.shape[0])).astype(float)
        self.radii = np.zeros(shape=self.Y.shape[0])
        self.radii.fill(r_st)
        for i in range(its):
            sample_size = 10
            # if i %(X.shape[0]/100) == 0:
            #     self.cand_hits.fill(0)
            if np.any(self.cand_hits):
                lrcoefs = 1 + 1. * self.cand_hits / self.cand_hits.max()
            else:
                pows = 1 + self.cand_hits
                lrcoefs = 1 + self.cand_hits
            self.radii = r_st * np.exp(-2. * i / (its)) / lrcoefs
            alpha = lr - i * lr * 1.0 / its  # * np.exp(-1.5*i/(its))
            alphas = alpha * (lrcoefs)
            # alphas = alphas *  redbase * lrcoefs
            # int(np.ceil(X.shape[0]*float(i/10+1)*10./its))
            xix = 0
            trinds = np.random.choice(X.shape[0], sample_size)
            for x in X[trinds]:
                et = timeit.default_timer() - self.start_time
                xix += 1
                bmu = pairwise_distances_argmin(np.array([x]), self.C, axis=1)[0]
                print '\r smoothing epoch %i / %i: %s : radius: %s : training_sample : %i / %i : time : %s ' % (
                i + 1, its, str(alphas[bmu]), str(self.radii[bmu]), xix, sample_size, str(et)),

                Ldist = grid_dists[bmu]
                # if (i+20 >= 2000):
                self.cand_hits[bmu] += 1
                neighborhood = np.where(Ldist < self.radii[bmu])[0]
                if neighborhood.shape[0] == 0:
                    neighborhood = np.argsort(Ldist)[:5]
                ''' we're going to fuck shit up with this'''

                Hdist = np.linalg.norm(self.C - self.C[bmu], axis=1)[neighborhood]
                if Hdist.max():
                    Hdist /= Hdist.max()
                thet_D = np.array([np.exp(-45.5 * Hdist ** 60)]).T
                thet_d = np.array([np.exp(-(12.5) * Ldist[neighborhood] ** 2 / np.max(
                    [self.radii[bmu], Ldist[neighborhood].max()]) ** 2)]).T
                w = np.array(self.C)[neighborhood]
                delts = np.array([alphas[neighborhood]]).T * ((x - w) * (thet_d)) -( self.wd * w * (
                1 - thet_D))  # *(1-np.exp(-2.5*(i/float(its)))))#*(1-thet_d))#*(1-np.exp(-2.5*(i/float(its)))))#*(i>=its-5))
                w += delts
                self.C[neighborhood] = w
                if i == its / 2 and xix == 1:
                    self.thet_vis_bundle['Y'] = self.Y
                    self.thet_vis_bundle['bmu'] = bmu
                    self.thet_vis_bundle['thet_d'] = thet_d
                    self.thet_vis_bundle['thet_D'] = thet_D
                    self.thet_vis_bundle['neighborhood'] = neighborhood

        print "\ntime for first smoothing iteration : ", (timeit.default_timer() - st), "\n"

        ########################################################################################################################

    def LMDS(self, X):
        r_st = 0.4
        radius = r_st

        grid = self.predict(X).astype(float)
        n = X.shape[0]*0.5
        its = 20
        it = 0
        st = timeit.default_timer()

        while it < its and radius > 0.001 and self.beta*np.exp(-7.5 * it**2  / its**2 ) > 0.001:# or n>1:
            radius *=0.95# r_st *  np.exp(- 9.0*it  / (50))
            et = timeit.default_timer() - self.start_time

            sys.stdout.write('\r LMDS iteration %i : radius : %s : beta : %s: time : %s ' % (it, str(radius), str(self.beta *np.exp(-7.5 * it**2  / its**2 )), str(et)))
              # np.exp(-10.0* iter  / self.iterations)
            Hdists = pairwise_distances(X)
            Ldists = pairwise_distances(grid)

            for i in range(X.shape[0]):
                grid -= grid.min(axis=0)
                grid /= grid.max(axis=0)
                Ldist = Ldists[i]
                Hdist = Hdists[i]

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
        print self.C.shape[0]
        return grid

