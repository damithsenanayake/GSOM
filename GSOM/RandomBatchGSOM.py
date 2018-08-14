
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
            ''' Conduct a PCA transformation of data if specified for better execution times. '''
            # if self.pca_ncomp:
            #     X = PCA(min(X.shape[0], X.shape[1], self.pca_ncomp)).fit_transform(X)
            its = 30
            st = timeit.default_timer()
            self.start_time = st
            self.grid = np.array([[i,j] for i in range(2) for j in range(int(2))])
            self.W = np.zeros((self.grid.shape[0], X.shape[1]))#np.random.RandomState(seed=5).random_sample(size=(self.grid.shape[0], X.shape[1]))#np.random.random(size=(self.grid.shape[0], X.shape[1]))
            self.W[:, :2] = self.grid
            self.W[:, :2] *= X[:, :2].max(axis=0) - X[:, :2].min(axis=0)
            self.W[:, :2] -= X[:, :2].mean(axis=0)
            self.hits = np.zeros(self.grid.shape[0]).astype(float)
            self.errors = np.zeros(self.grid.shape[0])

            rad_min = self.rad_min

            lambrad = np.log(rad_min * 1./ self.rst)
            min_lr = self.lrst*0.1#1. / its

            lambda_lr = -np.log(min_lr / self.lrst)
            fract_st = 1.
            min_fract = 2./X.shape[0]*np.pi*rad_min**2#0.01#


            lambda_fr = -np.log(min_fract/fract_st)
            for i in range(its):
                ''' Normalized Time Variable for the learning rules.'''
                ntime = i * 1. / max(its - 1, 1)
                sf = (self.sf_max-self.sf_min)*ntime + self.sf_min
                self.GT = -(X.shape[1]) * np.log(sf) * (X.max() - X.min())
                self.hits = np.zeros(self.grid.shape[0])
                r = self.rst*np.exp(lambrad * ntime)
                self.wd = 0.08 #* np.exp(-0.75*ntime)# * (0.5+0.5*(1-ntime))
                self.lr = self.lrst + (min_lr - self.lrst) * ntime**2 #*np.exp(-lambda_lr*ntime)
                xix = 0
                fract = fract_st*np.exp(-lambda_fr*ntime)
                batch_size = 1
                X_p = X#[np.random.permutation(X.shape[0])]
                for x in X_p:
                    ''' Training For Instances'''
                    bmu = pairwise_distances_argmin(np.array([x]), self.W, axis=1)[0]
                    self.errors[bmu] += np.linalg.norm(self.W[bmu] - x)
                    self.hits[bmu]+=1

                    ldist = np.linalg.norm(self.grid - self.grid[bmu], axis=1)
                    neighbors = np.where(ldist<=r)[0]#np.argsort(ldist)[:max(np.ceil((nix)), 5)]#np.where(ldist < r)[0]
                    dix = max(1,int(fract * self.W.shape[0]))
                    decayers = np.argsort((ldist))[:dix]

                    ''' ** coefficient to consider sinking to neighborhood! ** '''
                    ld = ldist[neighbors]/r
                    thetfunc = np.exp(-.5 * (ld)**2)#
                    theta_d = np.array([thetfunc]).T

                    self.W[neighbors]+= (x-self.W[neighbors])*theta_d*self.lr

                    self.errors[bmu]+= np.linalg.norm(self.W[bmu]-x)

                    ''' Growing When Necessary '''
                    if self.errors[bmu] > self.GT:
                        self.error_dist(bmu)


                    ''' Curvature Enforcement '''
                    g_center = self.W[bmu]
                    wd_coef = self.lr*self.wd
                    ''' ** coefficient to consider sinking to neighborhood! ** '''

                    sink = 1#np.product(np.linalg.norm(g_center-self.W, axis=1).argmin() - neighbors)

                    hdist = np.linalg.norm(self.W[decayers]-self.W[bmu], axis=1)
                    if hdist.max():
                        hdist /= hdist.max()

                    D =  np.array([hdist**2]).T#np.array([np.exp(-4.5*(1-hdist)**4)]).T
                    if sink:
                        self.W[decayers]-=(self.W[decayers]-g_center)*wd_coef*D
                    et = timeit.default_timer()-st

                    if xix % 500 == 0:
                        print (
                        '\riter %i : %i / %i : batch : %i :|G| = %i : n_neis :%i : LR: %.4f  QE: %.4f sink?: %s : wdFract: %.4f : wd_coef : %.4f' % (
                        i + 1, xix, X.shape[0], 1, self.W.shape[0], neighbors.shape[0], self.lr, self.errors.sum(),
                        str(dix), decayers.shape[0] * 1. / self.W.shape[0], np.mean(wd_coef))), ' time = %.2f' % (et),




                    # while self.errors.max()>self.GT:
                    #     self.error_dist(self.errors.argmax())
                    xix+=1

                self.prune_mid_training()

                # self.surface_tension()
                self.errors *= 0#self.hits/self.hits.max()
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

    def prune_mid_training(self):
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

        imm_neis = [up, left, down, right]

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
        Y = []
        # if self.pca_ncomp:
        #     X = PCA(min(X.shape[0], X.shape[1], self.pca_ncomp)).fit_transform(X)
        for x in X:
            Y.append(self.grid[np.argmin(np.linalg.norm(x-self.W, axis=1))])
        return np.array(Y)

