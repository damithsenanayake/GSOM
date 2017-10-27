import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as ds
from sklearn.manifold import TSNE
from protodsgsom import GSOM
print GSOM.__module__

X, t = ds.make_blobs(1500, random_state=20)

# Z, t2 = ds.make_s_curve(1000, random_state=1)
#
#
# W , t3 = ds.make_circles(500, random_state=1)
#
#
# Z = np.concatenate((Z, np.zeros((500,1))), axis=1)
# W = np.concatenate((W, np.zeros((500,1))), axis=1)
#
rand = np.random.RandomState(seed=10)
# # Z*=20
# Z += 10*rand.randn(3)
# #
# # W *= 20
# W -= 10*rand.randn(3)
#
# fig = plt.figure()
#
# ax = fig.add_subplot(111, projection = '3d')
#
# ax.scatter(X.T[0], X.T[1], X.T[2], c = t, cmap= plt.cm.Spectral)
# ax.scatter(Z.T[0], Z.T[1], Z.T[2] , c = t2/2.0, cmap = plt.cm.inferno)
# ax.scatter(W.T[0], W.T[1], Z.T[2] , c = t3/2., cmap= plt.cm.rainbow)
# plt.show()
#
# X = np.concatenate((X/20, Z/20), axis=0)
#
# X = np.concatenate((X, W), axis = 0)
gsom = GSOM()
# Y= GSOM().fit_transform(X, lr = 1.0, beta=0.00, sf=0.7, wd=0.02, fd=.9)#X,lr = 1.0, beta=0.0,sf=0.01, fd=0.75, wd=0.5)
Y = TSNE().fit_transform(X)
# Y = LocallyLinearEmbedding(n_neighbors=100).fit_transform(X)
print Y.shape

Ys = Y.T
# Yc = Y[1000:].T
# Ym = Y[1500:].T

plt.scatter(Ys[0], Ys[1], c = t, cmap = plt.cm.Spectral, alpha = 0.5)
# plt.scatter(Yc[0], Yc[1], c = t2, cmap=plt.cm.inferno, alpha = 0.5)
# plt.scatter(Ym[0], Ym[1], c = t3/2.0 , cmap=plt.cm.rainbow, alpha = 0.5)
plt.show()