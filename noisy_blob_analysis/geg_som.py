
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances
import timeit
from sklearn.decomposition import PCA
from sklearn.cluster import  KMeans
import matplotlib.pyplot as plt

class GEGSOM(object):

    def __init__(self, radius=8, min_rad=4, lrst=0.1, sf=0.9, fd=0.9, sd=0.04, cluster_spacing_factor=1., its=10,
                 labels=np.array([]), momentum=0., map_structure=8, neighbor_func='cut_gaussian', verbose=False):
        self.lrst = lrst
        self.its = its
        self.fd = fd
        self.wdst = sd
        self.hits = None
        self.W = None
        self.grid = None
        self.neighbor_setting = 'radial'
        self.rst = radius
        self.rad_min = min_rad
        self.sf_max = sf
        self._nei_func = neighbor_func
        self.plot = True
        self.recsf = cluster_spacing_factor
        if cluster_spacing_factor<1:
            self.csf = np.nan_to_num(1/(1-cluster_spacing_factor))
        else:
            self.csf = np.inf
        self.labels = labels
        self.last_hit = np.array([])
        self.momentum = momentum
        self.structure = map_structure
        self.n_low_neighbors = 0
        self.verbose = verbose



    def fit_transform(self, X):
        self.train_batch(X)
        return self.predict(X)

    def train_batch(self, X):
        try:
            its = self.its
            st = timeit.default_timer()
            self.start_time = st

            ''' Hexagonal initialization '''
            if self.structure == 'hex':
                self.n_low_neighbors = 6

            elif self.structure == 'square':
                self.n_low_neighbors = 4
            else:
                self.n_low_neighbors= int(self.structure)

            self.grid = np.zeros((self.n_low_neighbors+1, 2))

            for i in range(1,self.n_low_neighbors+1):
                angle = 2*np.pi*(i-1)/self.n_low_neighbors

                x = np.sin(angle)
                y = np.cos(angle)
                self.grid[i] = np.array([x, y])

            self.ages = np.zeros(self.grid.shape[0])

            self.W = np.zeros((self.grid.shape[0], X.shape[1]))
            self.W[:, :2] = self.grid
            self.W[:, :2] *= X[:, :2].max(axis=0) - X[:, :2].min(axis=0)
            self.W[:, :2] -= X[:, :2].mean(axis=0)
            self.hits = np.zeros(self.grid.shape[0]).astype(float)
            self.errors = np.zeros(self.grid.shape[0])
            self.lr = self.lrst
            rad_min = self.rad_min
            lambda_rad = np.log(rad_min*1./self.rst)
            lambda_lr = np.log(0.01)

            self.prevW = self.W*0

            for i in range(its):
                ''' Normalized Time Variable for the learning rules.'''
                ntime = i * 1. / max(its, 1)
                sf = self.sf_max
                self.GT = -np.sqrt(X.shape[1]) * np.log(sf)* (X.max() - X.min())
                r = self.rst *np.exp(lambda_rad * ntime)#- ntime * (self.rst - rad_min)
                self.r = r
                self.wd = self.wdst
                self.lr = self.lrst*np.exp(lambda_lr*ntime)#np.exp(lambda_lr*ntime)#self.lr*(1-ntime)#*(1-ntime)#*
                xix = 0
                try:
                    self.csf = 1/(1-self.recsf)
                except:
                    self.csf = np.inf

                self.errors *= 0
                self.ages *= 0
                for x in X:

                    ''' Training For Instances'''
                    bmu = pairwise_distances_argmin(np.array([x]), self.W, axis=1)
                    ldist = np.linalg.norm(self.grid - self.grid[bmu], axis=1)
                    hdist = np.linalg.norm(self.W - x, axis=1)
                    nix = np.where(ldist<=r)[0].shape[0]
                    dix = np.where(ldist<=r*self.csf)[0].shape[0]
                    decayers = np.argsort((ldist))#[:dix]#[:dix]#[:25*nix]#[:dix]
                    neighbors = np.argsort((ldist))[:nix]

                    ''' ** coefficient to consider sinking to neighborhood! ** '''
                    ld = ldist[neighbors]/r
                    thetfunc = self.neighbor_func(ld)#np.exp(-.5*ld**2)
                    theta_d = np.array([thetfunc]).T
                    delta_neis = (x-self.W[neighbors])*theta_d*self.lr#+ self.momentum *np.exp(-5*(1-ntime)**6)* self.prevW[neighbors]

                    ''' Gap  Enforcement '''
                    wd_coef = self.wd*self.lr#*(ntime)**.5
                    hdist = hdist[decayers]
                    hdist -= hdist.min()
                    hdist /= hdist.max()
                    D = np.exp(-7.*(1-hdist)**(2))
                    l = ldist[decayers]/ldist[decayers][min(dix, ldist[decayers].shape[0] - 1)]
                    d = np.exp( -(10*(1-self.recsf)) * l ** 60)
                    D *= d
                    pull = D-D.min()
                    pull /= pull.max()
                    pull = np.array([pull]).T
                    deltas =(self.momentum)*self.prevW#np.zeros(self.W.shape)
                    delta_dec=(x-self.W[decayers])*wd_coef*pull#*(i>1)
                    deltas[decayers] += delta_dec
                    deltas[neighbors] += delta_neis
                    self.W += deltas
                    # self.prevW[neighbors] = delta_neis
                    self.ages+=1
                    self.ages[neighbors] = 0
                    self.prevW = deltas
                    et = timeit.default_timer()-st
                    self.errors[bmu] += np.linalg.norm(self.W[bmu] - x)#**2
                    ''' Growing the map '''

                    while self.errors.max() > self.GT and i + 1 < its:
                        self.error_dist(self.errors.argmax())
                    self.prune_map(np.where(self.ages > 600))
                    if xix % 500 == 0 and self.verbose:
                        print (
                        '\riter %i of %i : %i / %i : batch : %i :|G| = %i : n_neis :%i : LR: %.4f  QE: %.4f sink?: %s : fd: %.4f : wd_coef : %.4f' % (
                        i + 1,its,  xix, X.shape[0], 1, self.W.shape[0], neighbors.shape[0], self.lr, self.errors.sum(),
                        str(self.csf), self.fd, np.mean(wd_coef))), ' time = %.2f' % (et),

                    xix+=1


                if self.labels.shape[0]:
                    fig = plt.figure(figsize=(5, 5))
                    Y = self.predict(X)
                    x, y = Y.T
                    plt.scatter(x, y, edgecolors='none', c=plt.cm.jet(self.labels *1./np.unique(self.labels).shape[0] ), alpha=0.5, s=15, marker='h')
                    # plt.show(block=False)
                    plt.savefig('./images/map_'+str(i)+'.png')
                    plt.close(fig)


            # self.smoothen()
        except KeyboardInterrupt:
            print self.W.shape[0]
            return

    def get_mid(self, decayers):

        return np.linalg.norm(self.grid[decayers] - self.grid[decayers].mean(axis=0), axis=1).argmin()

    def surface_tension(self, ixs):

        newW =self.W

        for i in ixs:

            neis = np.where(np.linalg.norm(self.grid[i]-self.grid, axis=1)-1<=0.00000001)

            newW[neis] = self.W[neis] + self.wd*(self.W[i]-self.W[neis])#.sum(axis=0)
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
        self.prevW = np.delete(self.prevW, ixs, axis=0)
        self.errors = np.delete(self.errors, ixs)
        self.grid = np.delete(self.grid, ixs,  axis=0)
        self.hits = np.delete(self.hits, ixs)
        self.ages = np.delete(self.ages, ixs)


    def smoothen(self):
        its = 0
        HDs = pairwise_distances(self.W, self.W)
        orig_lds = pairwise_distances(self.grid, self.grid)
        for i in range(its):
            lds = pairwise_distances(self.grid, self.grid)
            for ix in range(self.W.shape[0]):

                y = self.grid[ix]

                neis = np.where(orig_lds[ix]<=1)[0]

                D = HDs[ix][neis]
                d = lds[ix][neis]#np.linalg.norm(self.grid-self.grid[ix], axis=1)[neis]
                if D.max():
                    D/= D.max()
                # if d.max():
                #     d /= d.max()
                mu = np.exp(-.5*D**2)
                nu = ((1./(1+.5*d**2)**1.5))
                mu[mu ==0 ]=1
                pull = (mu - nu)*(nu)
                # push = 1- nu/mu


                # pull -= (1- 1./(1+d**2))

                force = ((self.grid[neis] - y) *np.array([pull]).T).sum(axis=0)

                self.grid[ix]+= 1*force

                print '\r %i / %i : %i / %i smoothen'%(ix, self.W.shape[0], i, its),

    def error_dist(self, g_node):

        imm_neis = np.zeros((self.n_low_neighbors, 2))

        for i in range(self.n_low_neighbors):
            angle = 2* np.pi / self.n_low_neighbors * i
            imm_neis[i] = self.grid[g_node]+np.array([np.sin(angle), np.cos(angle)])

        i = 0
        neierrors = np.zeros(self.n_low_neighbors)
        for nei in imm_neis:
            i += 1
            if self.point_exists(self.grid, nei):
                n_point = self.find_point(self.grid, nei)
                neierrors[i-1]=self.errors[n_point][0]
                self.errors[n_point] += (self.errors[n_point]) * self.fd
                self.errors[g_node]=0

        for i in range(self.n_low_neighbors):
            max_nei=np.argsort(neierrors)[i]
            nei = imm_neis[(max_nei+int(self.n_low_neighbors/2))%(self.n_low_neighbors)]
            if not self.point_exists(self.grid, nei):
                gdists_new = np.linalg.norm(nei - self.grid, axis=1)
                gdists_old = np.linalg.norm(self.grid - self.grid[g_node], axis=1)
                closest2_new = np.argsort(gdists_new)[:2]
                if np.any(abs(gdists_old[closest2_new] - 1)< .00000001):
                    w = self.W[closest2_new].mean(axis=0)
                else:
                    w = self.W[closest2_new[0]] * 2 - self.W[closest2_new[1]]

                self.W = np.append(self.W, np.array([w]), axis=0)
                self.errors = np.append(self.errors, 0.)
                self.grid = np.append(self.grid, np.array([nei]), axis=0)
                self.hits = np.append(self.hits, 0.)
                self.ages = np.append(self.ages, 0)
                self.errors[g_node] = 0
                self.prevW = np.append(self.prevW, np.array([w-self.W[g_node]]), axis=0)


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

    def neighbor_func(self, dists):
        if self._nei_func == 'bubble':
            return dists*0+1
        elif self._nei_func == 'cut_gaussian':
            return np.exp(-.5*dists**2)
        elif self._nei_func == 'gaussian':
            return np.exp(-3.5*dists**2)
        elif self._nei_func == 'epanechicov':
            return 1-dists**2
