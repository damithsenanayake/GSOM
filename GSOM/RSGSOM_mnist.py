import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
from MovingMap import  MovingMap
from RandomSampleGSOM import GSOM
from sklearn.preprocessing import normalize
from sklearn.manifold import MDS, TSNE, LocallyLinearEmbedding
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from SelfOrganizingSwarm import SelfOrganizingSwarm
from sklearn.decomposition import PCA
# from TSNE import TSNE
import timeit

print GSOM.__module__
fi = pd.read_csv('~/data/mnist_train.csv', header=None)
# test = pd.read_csv('../mnist_test.csv', header=None)
samples = 6000

dat =(np.array(fi)[:samples, 1:])#/255.0
dat = PCA(50).fit_transform(dat)
print dat.shape
# dat -= dat.min(axis=0)
# dat /= dat.max(axis=0)
labels = np.array(fi)[:samples, 0]
del fi
gc.collect()
# x, y = SelfOrganizingSwarm(iterations=10, alpha=1, beta=0.1, delta=0, theta=3.5).fit_transform(dat[:samples]).T
st = timeit.default_timer()
model = GSOM(lr=.1,  beta=0., sf=0.8, fd = .25, wd=.029)
# x, y = MovingMap(iterations=100, beta=1.5).fit_transform(dat[:samples]).T
Y= model.fit_transform(dat)
# Y = PCA(2).fit_transform(dat)

et = timeit.default_timer() - st
hrs = np.floor(et/3600)
mins = np.floor((et - 3600*hrs)/60)
secs = et-3600*hrs - 60* mins
print 'Total time : ', hrs, ':', mins, ':', secs
# Y= GSOM().fit_transform(dat, lr=.6,  beta=0., sf=0.9, fd = 1.9, wd=0.025)
# Y = TSNE().fit_transform(dat,perplexity=40)
# Y = PCA(2).fit_transform(dat)
# Y = TSNE(perplexity=40).fit_transform(dat)
# Y-= Y.min(axis=0)
# Y/= Y.max(axis=0)
x, y = Y.T
# x, y = MDS().fit_transform(dat[:samples]).T
fig = plt.figure(figsize=(5,10))
clusterer = KMeans(10)#DBSCAN(eps=0.025)
kl = clusterer.fit(Y).labels_
# print " lr=1,  beta=0.3, sf=0.9, fd = 0.9, wd=0.025"
print 'instances : ', samples
print 'ars :', adjusted_rand_score(labels, kl)
print 'ami :', adjusted_mutual_info_score(labels, kl)
plt.subplot(211)

np.savetxt('mnist_'+str(samples)+'.csv', np.concatenate((Y, np.array([labels]).T),axis=1))
plt.scatter(x, y, edgecolors='none',c=plt.cm.jet(labels/10.), alpha = 0.5, s = 15)
plt.subplot(212)
#
plt.scatter(x, y , edgecolors='none', c = kl, alpha = 0.5, s = 15)

plt.show()

''' Theta Analysis '''
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
#
#
# plt.figure(figsize=(5,10))
# plt.subplot(211)
# plt.scatter(x, y, s=15, alpha =0.001, c = 'grey')
# plt.scatter(x[neighborhood], y[neighborhood],c=reds)
# plt.title('input space')
#
# plt.subplot(212)
# plt.scatter(x, y, s=15, alpha =0.001, c = 'grey')
# plt.scatter(x[neighborhood], y[neighborhood], c=blues)
# plt.title('output space')
#
# plt.show()
