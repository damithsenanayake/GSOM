import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
from MovingMap import  MovingMap
from bgsom import GSOM
from sklearn.preprocessing import normalize
from sklearn.manifold import MDS, TSNE, LocallyLinearEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from SelfOrganizingSwarm import SelfOrganizingSwarm
from sklearn.decomposition import PCA

print GSOM.__module__
fi = pd.read_csv('~/data/mnist_train.csv', header=None)
# test = pd.read_csv('../mnist_test.csv', header=None)
samples = 500

dat =normalize(np.array(fi)[:samples, 1:])#/255.0
labels = np.array(fi)[:samples, 0]
del fi
gc.collect()
# x, y = SelfOrganizingSwarm(iterations=10, alpha=1, beta=0.1, delta=0, theta=3.5).fit_transform(dat[:samples]).T

# x, y = MovingMap(iterations=100, beta=1.5).fit_transform(dat[:samples]).T
Y= GSOM().fit_transform(dat,   lr=1,  beta=0.3, sf=0.99, fd = 1.9, wd=0.035)
# Y = TSNE(perplexity=40).fit_transform(dat)
x, y = Y.T
# x, y = MDS().fit_transform(dat[:samples]).T
fig = plt.figure()

kl = KMeans(10).fit(Y).labels_
# print " lr=1,  beta=0.3, sf=0.9, fd = 0.9, wd=0.025"
print 'instances : ', samples
print 'ars :', adjusted_rand_score(labels, kl)
print 'ami :', adjusted_mutual_info_score(labels, kl)
plt.subplot(211)

plt.scatter(x, y, edgecolors='none',c=plt.cm.jet(labels/10.0), alpha = 0.5, s = 15)
plt.subplot(212)

plt.scatter(x, y , edgecolors='none', c = 'grey', alpha = 0.5, s = 15)

plt.show()

