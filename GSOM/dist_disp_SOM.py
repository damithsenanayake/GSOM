
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances
import timeit
from sklearn.decomposition import PCA
from sklearn.cluster import  KMeans
import matplotlib.pyplot as plt

class GSOM(object):

    def __init__(self, min_rad=2.45, lrst=0.1, sf_min=0.3, sf_max=0.9, fd=0.15, radius=10,  sd=0.02, beta=0, PCA = 0, its=20,min_fract = 0.1, fract_start = 1., labels = np.array([])):
        self.lrst = lrst
        self.its = its
        self.fd = fd
        self.wdst = sd
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
        self.grid_shape = 'hex'
        self.min_fract = min_fract
        self.fract_start = fract_start
        self.plot = True
        self.labels = labels

    def fit_transform(self, X):
        self.train_batch(X)
        return self.predict(X)

    def train_batch(self, X):
        try:
            its = self.its
            st = timeit.default_timer()
            self.start_time = st

            ''' Hexagonal initialization '''
            if self.grid_shape == 'hex':
                self.grid = np.zeros((7, 2))

                for i in range(1,7):
                    angle = 2*np.pi*(i-1)/6

                    x = np.sin(angle)
                    y = np.cos(angle)
                    self.grid[i] = np.array([x, y])
            else:
                self.grid = np.array([[i, j ] for i in range(2) for j in range(2)])


            self.W = np.zeros((self.grid.shape[0], X.shape[1]))#np.random.RandomState(seed=5).random_sample(size=(self.grid.shape[0], X.shape[1]))#np.random.random(size=(self.grid.shape[0], X.shape[1]))
            self.W[:, :2] = self.grid
            self.W[:, :2] *= X[:, :2].max(axis=0) - X[:, :2].min(axis=0)
            self.W[:, :2] -= X[:, :2].mean(axis=0)
            self.hits = np.zeros(self.grid.shape[0]).astype(float)
            self.errors = np.zeros(self.grid.shape[0])
            self.lr = self.lrst
            rad_min = self.rad_min
            lambda_rad = np.log(rad_min*1./self.rst)
            lambda_lr = np.log(0.001)
            fract_st = self.fract_start
            fract_min = self.min_fract

            lambda_fr = -np.log(fract_min/fract_st)


            for i in range(its):
                ''' Normalized Time Variable for the learning rules.'''
                ntime = i * 1. / max(its, 1)
                sf = self.sf_max
                self.GT = -np.sqrt(X.shape[1]) * np.log(sf) * (X.max() - X.min())
                self.hits = np.zeros(self.grid.shape[0])
                r = self.rst *np.exp(lambda_rad * ntime)#- ntime * (self.rst - rad_min)
                self.wd = self.wdst#*(0.1+0.9*ntime)
                self.lr = self.lrst*(1-ntime)#np.exp(lambda_lr*ntime**2)#self.lr*(1-ntime)#*(1-ntime)#*
                xix = 0
                fract = 1.#fract_st*np.exp(-lambda_fr*ntime)#*(1-ntime)#
                self.errors *= 0
                batch_size = 1#int(50*ntime)+1
                n_batches = X.shape[0]/batch_size
                for b in range(n_batches):
                    bmus = pairwise_distances_argmin(X[b*batch_size:(b+1)*batch_size], self.W)
                    k =0
                    for x in X[b*batch_size:(b+1)*batch_size]:
                        ''' Training For Instances'''
                        bmu = bmus[k]
                        ldist = np.linalg.norm(self.grid - self.grid[bmu], axis=1)
                        nix = np.where(ldist<=r)[0].shape[0]
                        dix = max(nix,int(fract * self.W.shape[0]))
                        decayers = np.argsort((ldist))[:nix]
                        neighbors = decayers[:nix]
                        k+=1
                        ''' ** coefficient to consider sinking to neighborhood! ** '''
                        ld = ldist[neighbors]/r
                        thetfunc = np.exp(-.5* (ld)**2)
                        theta_d = np.array([thetfunc]).T
                        delta_neis = (x-self.W[neighbors])*theta_d*self.lr
                        ''' Gap  Enforcement '''
                        wd_coef = self.wd*self.lr#(1-ntime)**2
                        hdist = np.linalg.norm(self.W[decayers]-x, axis=1)
                        if ldist[decayers].max():
                            d = ldist[decayers]/ldist[decayers].max()#.max()
                        # d = (1+0.5*d**2)**-1
                        if hdist.max():
                            hdist /= (hdist.max())
                        D = hdist#np.exp(-0.5*(hdist)**2)


                        pull = d-D
                        # pull /= np.abs(pull).max()
                        # pull/=np.abs(pull).max()
                        # pull /= pull.max()
                        pull = np.array([pull]).T
                        delta_dec=(x-self.W[decayers])*wd_coef*pull
                        delta_dec[:neighbors.shape[0]] += delta_neis

                        self.W[decayers] += delta_dec
                        et = timeit.default_timer()-st

                        if xix % 500 == 0:
                            print (
                            '\riter %i of %i : %i / %i : batch : %i :|G| = %i : n_neis :%i : LR: %.4f  QE: %.4f sink?: %s : fd: %.4f : wd_coef : %.4f' % (
                            i + 1,its,  xix, X.shape[0], b, self.W.shape[0], neighbors.shape[0], self.lr, self.errors.sum(),
                            str(dix), self.fd, np.mean(wd_coef))), ' time = %.2f' % (et),
                        self.errors[bmu] += np.linalg.norm(self.W[bmu] - x)#**2

                        xix+=1
                        ''' Growing When Necessary '''
                        if self.errors[bmu] > self.GT and i+1 < its:
                            self.error_dist(bmu)

                self.prune_mid_training(X)

                if self.labels.shape[0]:
                    fig = plt.figure(figsize=(5, 5))
                    Y = self.predict(X)
                    x, y = Y.T
                    plt.scatter(x, y, edgecolors='none', c=plt.cm.jet(self.labels *1./np.unique(self.labels).shape[0] ), alpha=0.5, s=15, marker='h')
                    # plt.show(block=False)
                    plt.savefig('./images/map_'+str(i)+'_d.png')
                    plt.close(fig)


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

        imm_neis = np.zeros((6, 2))

        for i in range(6):
            angle = 2* np.pi / 6 * i
            imm_neis[i] = self.grid[g_node]+np.array([np.sin(angle), np.cos(angle)])

        for nei in imm_neis:
            if self.point_exists(self.grid, nei):
                n_point = self.find_point(self.grid, nei)
                self.errors[n_point] += (self.errors[g_node]-self.GT) * self.fd
            else:
                gdists_new = np.linalg.norm(nei - self.grid, axis=1)
                gdists_old = np.linalg.norm(self.grid - self.grid[g_node], axis=1)
                closest2_new = np.argsort(gdists_new)[:2]
                if np.any(abs(gdists_old[closest2_new] - 1)<0.00000001):
                    w = self.W[closest2_new].mean(axis=0)
                else:
                    w = self.W[closest2_new[0]] * 2 - self.W[closest2_new[1]]

                self.W = np.append(self.W, np.array([w]), axis=0)
                self.errors = np.append(self.errors, 0.)
                self.grid = np.append(self.grid, np.array([nei]), axis=0)
                self.hits = np.append(self.hits, 0.)
        self.errors[g_node] = 0*self.GT / 2


    def point_exists(self, space, point):
        dists = np.linalg.norm(space-point, axis=1)

        return dists.min()<0.0001# not np.linalg.norm(space-point, axis=1).all()

    def find_point(self, space, point):
        return np.where(np.linalg.norm(space-point, axis=1)<0.001)[0]

    def predict(self, X):
        Y = self.grid[pairwise_distances_argmin(X, self.W)]
        # if self.pca_ncomp:
        #     X = PCA(min(X.shape[0], X.shape[1], self.pca_ncomp)).fit_transform(X)

        return Y

