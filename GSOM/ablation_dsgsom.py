from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap, MDS
from TSOS import SelfOrganizingSwarm
from MovingMap import MovingMap
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ars
from sklearn.preprocessing import normalize
from cgsom import GSOM as GSOM
from Config_L import GSOM as DSGSOM

n_clusters = 10

rand = np.random.RandomState(1)
X, color = datasets.samples_generator.make_blobs(n_samples=3000, n_features=50, centers=n_clusters, cluster_std=3.5, random_state=20)

n_fac = .25
s_f = np.sqrt(X.var(axis=0))
noise = rand.random_sample(X.shape)
noise *= 2*s_f
noise -= s_f
noise /= n_fac
X += noise

configs = []

configs.append(GSOM())
configs.append(DSGSOM())

# X=  normalize(X)

# fig = plt.figure()

Y= DSGSOM(lr = .2, beta=0.4, sf=0.9, wd=0.08, fd=0.2).fit_transform(X)#X,lr = 1.0, beta=0.0,sf=0.01, fd=0.75, wd=0.5)
# fig = plt.figure()
# ax = Axes3D(fig)00
# ax.scatter(X.T[0], X.T.[1], X.T[2],c = color, alpha=0.5, edgecolors='none')
# plt.show()
# plt.subplot(211)
# ax = fig.add_subplot(211)
fig = plt.figure(num=None, figsize=(1.5,2.))
plt.scatter(Y.T[0], Y.T[1], s = 2, c = plt.cm.jet(color/(n_clusters*1.0)), edgecolors='none', alpha=0.375)


labs = KMeans(n_clusters).fit(Y).labels_

# plt.subplot(212)
# plt.scatter(Y.T[0], Y.T[1], s = 15, c =plt.cm.jet(labs/(n_clusters*1.0)), edgecolors='none', alpha=0.375)


print 'ars ', ars(color,labs)
print 'ami ', ami(color, labs)


#
# Y = Isomap().fit_transform(X)
# ax2 = fig.add_subplot(121)
# ax2.scatter(Y.T[0], Y.T[1], c = color, edgecolors='none', alpha=0.5)

plt.show()