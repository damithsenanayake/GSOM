
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances
import timeit


class GSOM(object):

    def __init__(self, n_neighbors=600, lrst=0.1, sf=0.9, fd=0.15, wd=0.02, beta=0):
        self.lrst = lrst
        self.sf = sf
        self.fd = fd
        self.wdst = wd
        self.beta = beta
        self.radst = np.sqrt(n_neighbors/2)
        self.n_neighbors = n_neighbors

    def fit_transform(self, X):
        self.train_batch(X)
        return self.LMDS(X)#self.predict(X)

    def train_batch(self, X):
        its = 15
        st = timeit.default_timer()
        self.start_time = st
        self.GT = -X.shape[1]* np.log(self.sf)* (X.max()-X.min())
        self.grid = np.array([[i,j] for i in range(2) for j in range(2)])
        self.W = np.random.random(size=(self.grid.shape[0], X.shape[1]))
        self.errors = np.zeros(self.grid.shape[0])
        self.lr=self.lrst
        is_trad = 0
        trad_its = 0
        for i in range(its):

            # Normalized Time Variable for the learning rules.

            ntime = i*1./max(its-1,1)

            self.hits = np.zeros(self.grid.shape[0])
            self.rad = self.radst #* np.exp(-.5*(i/float(its))**2)
            self.lr = self.lr #* np.exp(-.5 *ntime**2)
            self.wd = self.wdst
            '''Distribute Errors to propagate growth over the non hit areas'''
            while self.errors.max() >= self.GT:
                self.error_dist(self.errors.argmax())

            xix = 0
            fract =1.*np.exp(-2.5*ntime**2)# np.exp(-4.5*ntime**2)#np.exp(-5.5* ntime **2 ) #1-ntime#0.9**(i)#0.5*np.exp(-3.9*ntime**4)#(-ntime**2+1)*0.8#*
            is_trad = fract < min(fract, 0.*self.n_neighbors*1./self.W.shape[0])
            trad_its += is_trad
            # if i == 5:#trad_its:
            #     break
            for x in np.random.permutation(X):
                xix += 1
                ''' Training For Instances'''
                bmu = pairwise_distances_argmin(np.array([x]), self.W, axis=1)[0]
                self.hits[bmu]+=1
                r = self.rad
                dix = int(np.floor(self.grid.shape[0] * max(fract, self.n_neighbors*1./self.W.shape[0])))
                decayers = np.argsort(np.linalg.norm(self.grid[bmu] - self.grid, axis=1))[:dix]

                ldist = np.linalg.norm(self.grid - self.grid[bmu], axis=1)
                neighbors = np.where(ldist < r)
                theta_d = np.array([np.exp(-0.5 * (ldist[neighbors]/r)**2)]).T
                hdist = np.linalg.norm(self.W[decayers]-x, axis=1)
                hdist/=hdist.max()
                theta_D = np.array([1- np.exp(-20.5*hdist**6)]).T
                self.errors[bmu]+= np.linalg.norm(self.W[bmu]-x)
                self.W[neighbors]+= (x-self.W[neighbors])*theta_d*self.lr
                ''' Separating Weight Decay'''

                self.W[decayers]-=self.lr*self.wd*self.W[decayers]*theta_D#*(np.exp(-.5*(ntime)**2))
                et = timeit.default_timer()-st
                print ('\riter %i : %i / %i : |G| = %i : radius :%.4f : LR: %.4f  p(g): %.4f Rrad: %.2f : wdFract: %.4f'%(i+1,xix, X.shape[0], self.W.shape[0], r, self.lr,  np.exp(-8.*ntime**2), (self.n_neighbors*1./self.W.shape[0]), fract )),' time = %.2f'%(et),
                ''' Growing When Necessary '''
                if self.errors[bmu] >= self.GT:# and i<1.7*its:
                    self.error_dist(bmu)
        self.smoothen(X)

    def smoothen(self, X):
        its = 0
        print ''
        for i in range(its):
            for x in X:
                bmu = pairwise_distances_argmin(np.array([x]), self.W, axis=1)[0]
                neighbors = np.argsort(np.linalg.norm(self.grid[bmu]-self.grid, axis=1))[:5]
                self.W[neighbors] += (x-self.W[neighbors])*self.lr/2.
                print '\r %i / %i smoothen'%(i, its),

    def error_dist(self, g_node):
        up = self.grid[g_node] + np.array([0, 1])
        left = self.grid[g_node] + np.array([-1, 0])
        down = self.grid[g_node] + np.array([0, -1])
        right = self.grid[g_node] + np.array([1, 0])

        imm_neis = [up, left, down, right]

        for nei in imm_neis:
            if self.point_exists(self.grid, nei):
                n_point = self.find_point(self.grid, nei)
                self.errors[n_point] += self.errors[n_point] * self.fd
            else:
                gdists_new = np.linalg.norm(nei - self.grid, axis=1)
                gdists_old = np.linalg.norm(self.grid - self.grid[g_node], axis=1)
                closest2_new = np.argsort(gdists_new)[:2]
                if np.any(gdists_old[closest2_new] == 1):
                    w = self.W[closest2_new].mean(axis=0)
                else:
                    w = self.W[closest2_new[0]] * 2 - self.W[closest2_new[1]]

                self.W = np.append(self.W, np.array([w]), axis=0)
                self.errors = np.append(self.errors, 0)
                self.grid = np.append(self.grid, np.array([nei]), axis=0)
                self.hits = np.append(self.hits, 0)
        self.errors[g_node] = 0#self.GT / 2


    def point_exists(self, space, point):
        return not np.linalg.norm(space-point, axis=1).all()

    def find_point(self, space, point):
        return np.where(np.linalg.norm(space-point, axis=1)==0)[0]

    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.grid[np.argmin(np.linalg.norm(x-self.W, axis=1))])
        return np.array(Y)

    def LMDS(self, X):
        r_st = 0.4
        radius = r_st

        grid = self.predict(X).astype(float)
        n = X.shape[0]*0.5
        its = 20
        it = 0
        st = timeit.default_timer()

        while it < its and radius > 0.001 and self.beta*np.exp(-7.5 * it**2  / its**2 ) > 0.001:# or n>1:
            et = timeit.default_timer() - self.start_time
            radius = np.exp(-4.5*(it*1./its)**2)*r_st
            print '\r LMDS iteration %i : radius : %s : beta : %s: time : %s ' % (it, str(radius), str(self.beta *np.exp(-7.5 * it**2  / its**2 )), str(et)),
              # np.exp(-10.0* iter  / self.iterations)
            Hdists = pairwise_distances(X)
            Ldists = pairwise_distances(grid)

            for i in range(X.shape[0]):
                grid -= grid.min(axis=0)
                grid /= grid.max(axis=0)
                Ldist = Ldists[i]
                Hdist = Hdists[i]

                neighbors = np.where(Ldist < radius)[0]
                # neighbors = np.argsort(Ldist)[:100]
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

            it += 1
            n*=0.8
        print '\n LMDS time : ', timeit.default_timer() - st
        print '\n ', self.W.shape[0]
        return grid
