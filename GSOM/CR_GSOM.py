
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances
import timeit
from sklearn.decomposition import PCA
from sklearn.cluster import  KMeans

class GSOM(object):

    def __init__(self, min_rad=2.45, lrst=0.1, sf_min=0.3, sf_max=0.9, fd=0.15, radius=10,  wd=0.02, beta=0, PCA = 0):
        self.lrst = lrst
        self.fd = fd
        self.wdst = wd
        self.beta = beta
        self.pca_ncomp = PCA
        self.hits = None
        self.W = None
        self.grid = None
        self.neighbor_setting = 'radial'
        self.rst = radius
        self.rad_min = min_rad
        self.sf_min = sf_min
        self.sf_max = sf_max

    def fit_transform(self, X):
        self.train_batch(X)
        return self.predict(X)

    def train_batch(self, X):
        try:
            its = 20
            st = timeit.default_timer()
            self.start_time = st
            self.grid = np.array([[i,j] for i in range(2) for j in range(int(2))])
            self.W = np.zeros((self.grid.shape[0], X.shape[1]))#np.random.RandomState(seed=5).random_sample(size=(self.grid.shape[0], X.shape[1]))#np.random.random(size=(self.grid.shape[0], X.shape[1]))
            self.W[:, :2] = self.grid
            self.W[:, :2] *= X[:, :2].max(axis=0) - X[:, :2].min(axis=0)
            self.W[:, :2] -= X[:, :2].mean(axis=0)
            self.hits = np.zeros(self.grid.shape[0]).astype(float)
            self.errors = np.zeros(self.grid.shape[0])
            self.lr = self.lrst
            rad_min = self.rad_min

            lambrad = np.log(rad_min * 1./ self.rst)
            fract_st = 1.
            min_fract =.1# 0.05
            lrmin = 0.01*self.lrst#*1./its
            lambda_lr = np.log(lrmin/self.lrst)


            lambda_fr = -np.log(min_fract/fract_st)
            for i in range(its):
                ''' Normalized Time Variable for the learning rules.'''
                ntime = i * 1. / max(its, 1)
                sf = self.sf_max
                self.GT = -np.sqrt(X.shape[1]) * np.log(sf) * (X.max() - X.min())
                self.hits = np.zeros(self.grid.shape[0])
                r = self.rst - ntime * (self.rst - rad_min)
                self.wd = 0.04
                self.lr = self.lrst*np.exp(lambda_lr*ntime)
                xix = 0
                fract = fract_st*np.exp(-lambda_fr*ntime)
                self.errors *= 0
                batch_size = int(50*ntime)+1
                n_batches = X.shape[0]/batch_size

                for b in range(n_batches):
                    bmus = pairwise_distances_argmin(X[b*batch_size:(b+1)*batch_size], self.W)
                    k =0
                    for x in X[b*batch_size:(b+1)*batch_size]:
                        ''' Training For Instances'''
                        bmu = bmus[k]
                        ldist = np.linalg.norm(self.grid - self.grid[bmu], axis=1)
                        nix = np.where(ldist<=r)[0].shape[0]
                        dix = max(2*nix,int(fract * self.W.shape[0]))
                        decayers = np.argsort((ldist))[:dix]
                        neighbors = decayers[:nix]
                        k+=1
                        ''' ** coefficient to consider sinking to neighborhood! ** '''
                        ld = ldist[neighbors]/r
                        thetfunc = np.exp(-0.5* (ld))#(1+ld**2)**-1#
                        theta_d = np.array([thetfunc]).T
                        delta_neis = (x-self.W[neighbors])*theta_d*self.lr
                        ''' Gap  Enforcement '''
                        wd_coef = self.lr*self.wd#*np.exp(-.5*ntime**4)*(1-ntime)**.5#*(fract<0.5)
                        hdist = np.linalg.norm(self.W[decayers]-x, axis=1)
                        hdist -= hdist.min()
                        hdist /= hdist.max()
                        dist = ldist[decayers]/ldist[decayers].max()
                        d = (1+dist**6)**-1
                        pull = 1-d
                        pull = np.array([pull]).T
                        delta_dec=(x-self.W[decayers])*wd_coef*pull
                        delta_dec[:neighbors.shape[0]] += delta_neis

                        self.W[decayers] += delta_dec
                        et = timeit.default_timer()-st

                        if xix % 500 == 0:
                            print (
                            '\riter %i of %i : %i / %i : batch : %i :|G| = %i : n_neis :%i : LR: %.4f  QE: %.4f sink?: %s : wdFract: %.4f : wd_coef : %.4f' % (
                            i + 1,its,  xix, X.shape[0], b, self.W.shape[0], neighbors.shape[0], self.lr, self.errors.sum(),
                            str(dix), decayers.shape[0] * 1. / self.W.shape[0], np.mean(wd_coef))), ' time = %.2f' % (et),
                        self.errors[bmu] += np.linalg.norm(self.W[bmu] - x)#**2

                        xix+=1
                    ''' Growing When Necessary '''
                    while self.errors.max() > self.GT:
                        self.error_dist(self.errors.argmax())

                self.prune_mid_training(X)
            self.smoothen(X)
        except KeyboardInterrupt:
            return

    def get_mid(self, decayers):

        return np.linalg.norm(self.grid[decayers] - self.grid[decayers].mean(axis=0), axis=1).argmin()

    def surface_tension(self):

        newW = np.zeros(self.W.shape)

        for i in range(self.W.shape[0]):

            neis = np.where(np.linalg.norm(self.grid[i]-self.grid, axis=1)==1)

            newW[i] = self.W[i] + (self.W[neis]-self.W[i]).sum(axis=0)
        self.W = newW

    def prune_mid_training(self, X):
        self.hits*=0
        bmus = pairwise_distances_argmin(X, self.W)

        for b in bmus:
            self.hits[b]+=1
        ''' Moving Average Filter to identify contiguous regions in the map '''
        self.mean_filter(1)

        ''' Prune nodes in the non-continguous regions of the map to shave of training time '''
        self.prune_map(np.where(self.hits == 0)[0])

    def mean_filter(self, degree=1):
        for i in range(degree):
            self.new_hits = np.zeros(self.hits.shape)
            for i in range(self.hits.shape[0]):
                neighbors = np.where((np.linalg.norm(self.grid[i]-self.grid, axis=1)<=1))[0]
                self.new_hits[i]= self.hits[neighbors].mean()
            self.hits =self.new_hits

    def prune_map(self, ixs):
        self.W = np.delete(self.W, ixs, axis=0)
        self.errors = np.delete(self.errors, ixs)
        self.grid = np.delete(self.grid, ixs,  axis=0)
        self.hits = np.delete(self.hits, ixs)


    def smoothen(self, X):
        its = 0
        print ''
        for i in range(its):
            for x in X:
                bmu = pairwise_distances_argmin(np.array([x]), self.W, axis=1)[0]
                ldists = np.linalg.norm(self.grid[bmu]-self.grid, axis=1)
                neighbors = np.argsort(ldists)[:10]
                hs = np.exp(-0.5*(ldists[neighbors]/ldists[neighbors].max())**2)
                self.W[neighbors] += (x-self.W[neighbors])*self.lr*np.array([hs]).T
                print '\r %i / %i smoothen'%(i, its),

    def error_dist(self, g_node):
        up = self.grid[g_node] + np.array([0, 1])
        left = self.grid[g_node] + np.array([-1, 0])
        down = self.grid[g_node] + np.array([0, -1])
        right = self.grid[g_node] + np.array([1, 0])
        imm_neis = [up, left, down, right]#, lu, ld, ru, rd]

        for nei in imm_neis:
            if self.point_exists(self.grid, nei):
                n_point = self.find_point(self.grid, nei)
                self.errors[n_point] += (self.errors[g_node]-self.GT) * self.fd
            else:
                gdists_new = np.linalg.norm(nei - self.grid, axis=1)
                gdists_old = np.linalg.norm(self.grid - self.grid[g_node], axis=1)
                closest2_new = np.argsort(gdists_new)[:2]
                if np.any(gdists_old[closest2_new] == 1):
                    w = self.W[closest2_new].mean(axis=0)
                else:
                    w = self.W[closest2_new[0]] * 2 - self.W[closest2_new[1]]

                self.W = np.append(self.W, np.array([w]), axis=0)
                self.errors = np.append(self.errors, 0.)
                self.grid = np.append(self.grid, np.array([nei]), axis=0)
                self.hits = np.append(self.hits, 0.)
        self.errors[g_node] = 0*self.GT / 2


    def point_exists(self, space, point):
        return not np.linalg.norm(space-point, axis=1).all()

    def find_point(self, space, point):
        return np.where(np.linalg.norm(space-point, axis=1)==0)[0]

    def predict(self, X):
        Y = self.grid[pairwise_distances_argmin(X, self.W)]
        # if self.pca_ncomp:
        #     X = PCA(min(X.shape[0], X.shape[1], self.pca_ncomp)).fit_transform(X)

        return Y

