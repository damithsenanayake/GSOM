import numpy as np
import pandas as pd
from WGSOM import  GSOM
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from SelfOrganizingSwarm import SelfOrganizingSwarm
from MovingMap import MovingMap
from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans
from sklearn.metrics import normalized_mutual_info_score
from bgsom import GSOM
data = pd.read_csv("~/data/2D3C_RAW.csv", header=None)

X = np.array(data)


# X = normalize(X, axis = 0)

# add noise

# noise = np.random.randn(300, 90)*2
# X = np.append(X,noise, axis = 1)

X += abs(X.min())
X /= X.max()

indices = np.random.permutation(X.shape[0])
# gsom = GSOM(dims=92, hid = 2, sf = 0.8, fd = 0.9, max_nodes = 2000, min_nodes = 10, radius=60, X = X, gaussian=True, nei=True)
#
# for i in range(40):
#     print "\nbatch ", (i+1)
#     gsom.train_batch(X[indices], iterations=1, lr = 0.25*np.exp(-i/10), prune=True)
#
# # gsom.prune()
#
#
# grid, hits = gsom.predict(X)
# x, y = grid.T

# Y = SelfOrganizingSwarm(iterations=25, alpha=1, beta=0.5, delta= 0.01 , theta=3.5).fit_transform(X)
# Y = MovingMap(iterations=25, beta=1).fit_transform(X)

Y = GSOM().fit_transform(X, beta=0.5, lr=0.5)

x, y = Y.T
# plt.scatter(X.T[0], X.T[1])
# plt.show()
dbscan = DBSCAN(eps=0.01).fit(Y)
kmeans = KMeans(n_clusters=3).fit(Y)
print "\n"
print np.unique(dbscan.labels_).shape

print "\nnmi:", normalized_mutual_info_score(np.array(range(1500))/500, kmeans.labels_)

plt.scatter(x[:500], y[:500], color = 'red',alpha = 0.5)
plt.scatter(x[500:1000], y[500:1000], color = 'blue', alpha=0.5)
plt.scatter(x[1000:], y[1000:], color = 'green', alpha=0.5)

# for i, j, t in zip(x, y, kmeans.labels_):
#     plt.text(i, j, t, color=plt.cm.Set1(t/3.0), alpha = 0.5)
plt.show()
# x, y = PCA(2).fit_transform(X).T
# plt.scatter(x[:500], y[:500], color = 'red',marker = 4)
# plt.scatter(x[500:1000], y[500:1000], color = 'blue', marker = 5)
# plt.scatter(x[1000:], y[1000:], color = 'green', marker = 6)
# plt.show()
#
# x,y = TSNE(2).fit_transform(X).T
# plt.scatter(x[:100], y[:100], color = 'red',marker = 4)
# plt.scatter(x[100:200], y[100:200], color = 'blue', marker = 5)
# plt.scatter(x[200:], y[200:], color = 'green', marker = 6)
# plt.show()

#
#
# plt.scatter(x[:100], y[:100], color = 'red')
# plt.show()
#
# plt.scatter(x[100:200], y[100:200], color = 'blue')
# plt.show()
#
# plt.scatter(x[200:], y[200:], color = 'green')
# plt.show()