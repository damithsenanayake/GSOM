
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances
import timeit
from sklearn.decomposition import PCA
from sklearn.cluster import  KMeans

class GSOM(object):

    def __init__(self, n_neighbors=600, lrst=0.1, sf=0.9, fd=0.15, wd=0.02, beta=0, PCA = 0):
        self.lrst = lrst
        self.sf = sf
        self.fd = fd
        self.wdst = wd
        self.beta = beta
        self.radst = np.sqrt(n_neighbors/2)
        self.n_neighbors = n_neighbors
        self.pca_ncomp = PCA
        self.hits = None
        self.W = None
        self.grid = None
        self.neighbor_setting = 'radial'

    def fit_transform(self, X):
        self.train_batch(X)
        return self.predict(X)

    def train_batch(self, X):

        ''' Conduct a PCA transformation of data if specified for better execution times. '''
        # if self.pca_ncomp:
        #     X = PCA(min(X.shape[0], X.shape[1], self.pca_ncomp)).fit_transform(X)
        its = 30
        st = timeit.default_timer()
        self.start_time = st
        self.GT = -(X.shape[1])* np.log(self.sf)*(X.max()-X.min())
        self.grid = np.array([[i,j] for i in range(2) for j in range(int(2))])
        self.W = np.zeros(shape=(self.grid.shape[0], X.shape[1]))
        self.lr=self.lrst
        self.hits = np.zeros(self.grid.shape[0]).astype(float)
        self.errors = np.zeros(self.grid.shape[0])
        min_lr = 0.05#1. / its

        lambda_lr = -np.log(min_lr / self.lrst)
        fract_st = 1.
        min_fract = 0.1

        self.wdst = 0.08
        self.wden = 0.04

        lambda_fr = -np.log(min_fract/fract_st)

        min_neis = 10.

        lambda_wd = -np.log(self.wden/self.wdst)

        for i in range(its):
            ''' Normalized Time Variable for the learning rules.'''

            ntime = i * 1. / max(its - 1, 1)
            # if i==5:
            #     break

            if self.hits.sum():
                self.prune_mid_training()
            self.wd = self.wdst*np.exp(-lambda_wd * (1-ntime))
            self.hits = np.zeros(self.grid.shape[0])
            rad_lambda = - np.log(min_neis/self.n_neighbors)
            self.rad = np.sqrt(0.5*self.n_neighbors * np.exp(-rad_lambda * ntime ))#self.radst*np.exp(-rad_lambda*ntime)


            self.lr = self.lrst*np.exp(-lambda_lr*ntime)#(1-ntime)
            xix = 0
            fract =fract_st*np.exp(-lambda_fr*ntime)#**0.5#(1-ntime + (ntime**6/20))#(1-ntime)#+(ntime)**2/8)#0.9**i#np.exp( - 3.5 * (ntime))

            cent_fract = fract#**0.5#0.5# * np.exp(lambda_cf * ntime)#4*fract#(1- cent_fract_st)*(1-ntime) + cent_fract_st
            r = self.rad
            while self.errors.max() >= self.GT:
                self.error_dist(self.errors.argmax())
            for x in X:
                vis = 0
                if xix == 5999 and i == its-1:
                    vis = 1
                ''' Training For Instances'''
                try:
                    bmu = pairwise_distances_argmin(np.array([x]), self.W, axis=1)[0]
                except:
                    pass
                self.hits[bmu]+=1

                ldist = np.linalg.norm(self.grid - self.grid[bmu], axis=1)
                neighbors = np.where(ldist < r)[0]
                dix = int(fract * self.W.shape[0])
                cix = int(cent_fract * self.W.shape[0])
                decayers = np.argsort((ldist))[:dix]
                hemis = np.argsort(ldist)[:max(cix,2)]
                theta_d = np.array([np.exp(-.5 * (ldist[neighbors]/r)**2)]).T
                self.W[neighbors]+= (x-self.W[neighbors])*theta_d*self.lr

                ''' Curvature Enforcement '''
                hdist = np.linalg.norm(self.W[decayers]-self.W[bmu], axis=1)#
                if hdist.shape[0] and not(hdist.max()==0):
                    hdist -= hdist.min()
                    if hdist.max():
                        hdist/=hdist.max()


                theta_D = np.array([np.exp(-6.5*(1-hdist)**1)]).T
                wd_coef = self.lr*(self.wd)*theta_D#*np.exp(-0.75*(ntime))
                # wd_coef *= (its-i<=ncuriters)
                g_center = self.W[self.errors[neighbors].argmin()]#self.W[decayers].mean(axis=0)#self.W[self.hits[neighbors].argmin()]#kcenters[klabels[xix]]

                self.W[decayers]-=(self.W[decayers]-g_center)*wd_coef


                # decayers_distances = np.linalg.norm(self.W[decayers] - g_center, axis=1)

                # dist_ratio = 1- np.linalg.norm(self.W[hemis].mean(axis=0) - self.W[bmu])/ decayers_distances.max()
                # wd_ratio = np.exp( -4.5 * dist_ratio **6 )
                #
                # wd_coef *= wd_ratio

                self.errors[bmu]+= np.linalg.norm(self.W[bmu]-x)

                et = timeit.default_timer()-st
                if xix%500==0:
                    print ('\riter %i : %i / %i : |G| = %i : radius :%.4f : LR: %.4f  QE: %.4f Rrad: %.2f : wdFract: %.4f : wd_coef : %.4f'%(i+1,xix, X.shape[0], self.W.shape[0], r, self.lr,  self.errors.sum(), (self.n_neighbors*1./self.W.shape[0]), decayers.shape[0]*1./self.W.shape[0], np.mean(wd_coef) )),' time = %.2f'%(et),
                ''' Growing When Necessary '''


                if self.errors[bmu] > self.GT:
                    self.error_dist(bmu)

                if vis:
                    self.bmu = bmu
                    self.decayers = decayers
                    self.hemis  = hemis
                    self.undelgrid = self.grid
                    self.mid = np.argsort(np.linalg.norm(self.W[hemis].mean(axis=0)-self.W, axis=1))[0]
                    self.abcent = np.argsort(np.linalg.norm(self.W.mean(axis=0)-self.W, axis=1))[0]
                    self.theta_D = theta_D
        # self.prune_mid_training()
                xix += 1
        self.smoothen(X)

    def prune_mid_training(self):
        ''' Moving Average Filter to identify contiguous regions in the map '''
        self.mean_filter()

        ''' Prune nodes in the non-continguous regions of the map to shave of training time '''
        self.prune_map(np.where(self.hits == 0)[0])

    def mean_filter(self):
        self.new_hits = np.zeros(self.hits.shape)
        for i in range(self.hits.shape[0]):
            neighbors = np.argsort(np.linalg.norm(self.grid[i]-self.grid, axis=1))[:5]
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
                self.errors = np.append(self.errors, 0.)
                self.grid = np.append(self.grid, np.array([nei]), axis=0)
                self.hits = np.append(self.hits, 0.)
        self.errors[g_node] = 0#self.GT / 2


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

