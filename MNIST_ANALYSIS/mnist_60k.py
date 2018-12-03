import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
from geg_som import GEGSOM
from sklearn.preprocessing import normalize
from sklearn.manifold import MDS, TSNE, LocallyLinearEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.decomposition import PCA
import timeit
import warnings
from umap import UMAP

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import md5, sha

print GEGSOM.__module__
fi = pd.read_csv('~/data/mnist_train.csv', header=None)
# test = pd.read_csv('../mnist_test.csv', header=None)
samples = 6000

dat = (np.array(fi)[:samples, 1:])  # /255.0
dat = PCA(30, random_state=1).fit_transform(dat)

labels = np.array(fi)[:samples, 0]

del fi
print dat.shape

gc.collect()
st = timeit.default_timer()
model = GEGSOM(lrst=.2, sf=0.9, fd=.8, radius=8, min_rad=4, cluster_spacing_factor=1., sd=.03, its=25, labels=labels,
               momentum=.0, map_structure=6)

Y = model.fit_transform(dat)

et = timeit.default_timer() - st
hrs = np.floor(et / 3600)
mins = np.floor((et - 3600 * hrs) / 60)
secs = et - 3600 * hrs - 60 * mins
print 'Total time : ', hrs, ':', mins, ':', secs

x, y = Y.T

clusterer = KMeans(10)
kl = clusterer.fit(Y).labels_
print 'instances : ', samples
print 'ars :', adjusted_rand_score(labels, kl)
print 'ami :', adjusted_mutual_info_score(labels, kl)
fig = plt.figure(figsize=(5, 10))
plt.subplot(211)

np.savetxt('mnist_' + str(samples) + '.csv', np.concatenate((Y, np.array([labels]).T), axis=1))
plt.scatter(x, y, edgecolors='none', c=plt.cm.jet(labels / 10.), alpha=0.5, s=15, marker='h')
plt.subplot(212)
#
plt.scatter(x, y, edgecolors='none', c=plt.cm.jet(kl / 10.), alpha=0.5, s=15, marker='h')

plt.show()
