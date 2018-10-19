import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
from MovingMap import  MovingMap
from HEXGSOM import GSOM
from sklearn.preprocessing import normalize
from sklearn.manifold import MDS, TSNE, LocallyLinearEmbedding
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from SelfOrganizingSwarm import SelfOrganizingSwarm
from sklearn.decomposition import PCA
# from TSNE import TSNE
import timeit
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import md5, sha

print GSOM.__module__
fi = pd.read_csv('~/data/fashionmnist/fashion-mnist_train.csv')
# test = pd.read_csv('../mnist_test.csv', header=None)
samples = 20000

dat =(np.array(fi)[:samples, 1:]).astype(float)#/255.0
dat = PCA(10, random_state=1).fit_transform(dat)
# dat -= dat.min()
# dat /= dat.max()
labels = np.array(fi)[:samples, 0].astype(int)

# dat = dat[(labels ==5) | (labels == 3)]
# labels = labels[(labels == 5)|( labels == 3)]
del fi
print dat.shape

gc.collect()
# x, y = SelfOrganizingSwarm(iterations=10, alpha=1, beta=0.1, delta=0, theta=3.5).fit_transform(dat[:samples]).T
st = timeit.default_timer()
model = GSOM(lrst=.05, sf_max=0.9, fd = .01, radius=4, min_rad = 4, sd=0.01, its=20, min_fract=1., fract_start=1., labels = labels)

# x, y = MovingMap(iterations=100, beta=1.5).fit_transform(dat[:samples]).T
Y= model.fit_transform(dat)
# Y = PCA(2).fit_transform(dat)

YS = model.grid
WS = model.W

ds = np.linalg.norm(YS-YS[2], axis=1)
order = ds.argsort()
DS = np.linalg.norm(WS-WS[2], axis=1)
ds/=ds.max()
DS/=DS.max()

plt.plot(ds[order], DS[order])
plt.show(block=False)

et = timeit.default_timer() - st
hrs = np.floor(et/3600)
mins = np.floor((et - 3600*hrs)/60)
secs = et-3600*hrs - 60* mins
print 'Total time : ', hrs, ':', mins, ':', secs
# Y = TSNE().fit_transform(dat,perplexity=40)
# Y = PCA(2).fit_transform(dat)
# Y = TSNE(perplexity=40).fit_transform(dat)
# Y-= Y.min(axis=0)
# Y/= Y.max(axis=0)
x, y = Y.T
# x, y = MDS().fit_transform(dat[:samples]).T
# fig = plt.figure(figsize=(5,10))
clusterer = KMeans(10)#DBSCAN(eps=0.025)
kl = clusterer.fit(Y).labels_
# print " lr=1,  beta=0.3, sf=0.9, fd = 0.9, wd=0.025"
print 'instances : ', samples
print 'ars :', adjusted_rand_score(labels, kl)
print 'ami :', adjusted_mutual_info_score(labels, kl)
fig = plt.figure(figsize=(5, 10))
plt.subplot(211)

np.savetxt('mnist_'+str(samples)+'.csv', np.concatenate((Y, np.array([labels]).T),axis=1))
plt.scatter(x, y, edgecolors='none',c=plt.cm.jet(labels/10.), alpha = 0.5, s = 15, marker='h')
plt.subplot(212)
#
plt.scatter(x, y, edgecolors='none',c=plt.cm.jet(kl/10.), alpha = 0.5, s = 15, marker='h')
#
# plt.show(block=False)
#
#
# fig =plt.figure()
#



# #plt.cm.gist_rainbow(model.hits/float(model.hits.max()))
# plt.subplot(211)
# plt.scatter(model.grid.T[0], model.grid.T[1] , edgecolors='none', c = 'black', alpha = 0.1, s = 8, marker='+')
#
# x, y = model.undelgrid[model.hemis].T
#
# plt.scatter(x, y, edgecolors='none', c = 'blue', alpha=0.5, s = 20)
#
# x, y = model.undelgrid[model.mid]
# plt.scatter(x, y, edgecolors='none', c = 'green', alpha = 1., s = 20)
#
# x, y = model.undelgrid[model.decayers].T
# plt.scatter(x, y, edgecolors= 'none', c = 'red', alpha = 0.5, s = 10)
#
# x, y = model.undelgrid[model.bmu]
#
# plt.scatter(x, y, edgecolors='none', c='black', alpha=0.8, s = 10)
#
# x, y = model.undelgrid[model.abcent]
#
# plt.scatter(x, y, edgecolors='none', c='orange', alpha=0.8, s = 20, marker= 'x')
#
# plt.subplot(212)
#
# plt.scatter(model.grid.T[0], model.grid.T[1] , edgecolors='none', c = model.errors, cmap= plt.cm.gist_rainbow, alpha = 0.6, s = 20)
# x, y = model.undelgrid[model.decayers].T
#
# colors = np.zeros((x.shape[0],4))
#
# colors[:, 3] = model.theta_D.T
#
# plt.scatter(x, y, edgecolors= 'none', c = colors, s = 40)

plt.show()

# fig =plt.figure()
# plt.scatter(model.Y.T[0], model.Y.T[1] , edgecolors='none', c = plt.cm.gist_rainbow(model.radii/float(model.radii.max())), alpha = 0.5, s = 15)
# plt.show(block=False)

# ''' Theta Analysis '''
#
# bundle = model.thet_vis_bundle
#
# x, y = bundle['Y'].T
# thetD = bundle['thet_D']
# thetd = bundle['thet_d']
# neighborhood = bundle['neighborhood']
# bmu = bundle['bmu']
#
# rgbas = np.zeros((neighborhood.shape[0], 4))
#
# reds = rgbas + np.array([1, 0, 0,0])
# blues = rgbas + np.array([0, 1, 0, 0])
#
# reds[:, 3]= thetD.T[0]
# blues[:, 3]= thetd.T[0]
# #
#
# plt.figure(figsize=(10,10))
# # plt.subplot(211)
# plt.scatter(x, y, s=15, alpha =0.001, c = 'grey')
# plt.scatter(x[neighborhood], y[neighborhood],c=reds, edgecolors='none')
# plt.title('input space')
#
# # plt.subplot(212)
# # plt.scatter(x, y, s=15, alpha =0.001, c = 'grey')
# plt.scatter(x[neighborhood], y[neighborhood], c=blues, edgecolors='none')
# plt.title('output space')
#
# plt.show()
