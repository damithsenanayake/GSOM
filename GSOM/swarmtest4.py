import numpy as np
import sklearn.datasets as ds
from SelfOrganizingSwarm import SelfOrganizingSwarm
from MovingMap import  MovingMap
from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, scale
from bgsom import  GSOM
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN

from sklearn.metrics import normalized_mutual_info_score

X, labels = ds.make_moons(n_samples=2000, random_state=8)#(n_samples=1500, noise= 0.01)

noise = np.random.randn(X.shape[0], 3)*0.45

# plt.scatter(X.T[0], X.T[1], c= y, alpha=0.5)
# plt.show()


X = np.append(X,noise, axis=1)

X = normalize(X, axis=1)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.scatter(X.T[0], X.T[1], X.T[2],c = labels, alpha=0.5, edgecolors='none')
# plt.show()


# Y = SelfOrganizingSwarm(iterations=25, alpha=1, beta=0.0, delta=0.0 , theta=3.5).fit_transform(X)
# Y = TSNE(2).fit_transform(X)
# Y = MovingMap(iterations=50, beta=1).fit_transform(X)
Y = GSOM().fit_transform( X, lr=1.0 , beta=0.25, sf=0.3, fd = 0.9, wd=0.03)

af = DBSCAN(eps=0.1).fit(Y)

c = af.labels_

x, y = Y.T
plt.subplot(211)
print '\n'+str(normalized_mutual_info_score(labels, c))
plt.scatter(x,y, c= 'gray', alpha=0.5, edgecolors='none', s = 15)
plt.subplot(212)
labels -= labels.min()
plt.scatter(x, y, c=plt.cm.jet(labels.astype(float)/labels.max()), alpha = 0.5, edgecolors='none', s = 15)
plt.show()
