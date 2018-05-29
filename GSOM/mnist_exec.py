# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
from MovingMap import  MovingMap
from bgsom import GSOM
from sklearn.preprocessing import normalize
# from sklearn.manifold import MDS, TSNE, LocallyLinearEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from SelfOrganizingSwarm import SelfOrganizingSwarm
from sklearn.decomposition import PCA
# from TSNE import TSNE

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
model = GSOM(lr=.1,  beta=0., sf=0.7, fd = .18, wd=.04)
# x, y = MovingMap(iterations=100, beta=1.5).fit_transform(dat[:samples]).T
Y= model.fit_transform(dat)
x, y = Y.T
# x, y = MDS().fit_transform(dat[:samples]).T
# fig = plt.figure()

kl = KMeans(10).fit(Y).labels_
# print " lr=1,  beta=0.3, sf=0.9, fd = 0.9, wd=0.025"
print 'instances : ', samples
print 'ars :', adjusted_rand_score(labels, kl)
print 'ami :', adjusted_mutual_info_score(labels, kl)
# plt.subplot(211)

np.savetxt('mnist_'+str(samples)+'.csv', np.concatenate((Y, np.array([labels]).T),axis=1))
# plt.scatter(x, y, edgecolors='none',c=plt.cm.jet(labels/10.0), alpha = 0.5, s = 15)
# plt.subplot(212)
#
# plt.scatter(x, y , edgecolors='none', c = 'grey', alpha = 0.5, s = 15)

# plt.show()

