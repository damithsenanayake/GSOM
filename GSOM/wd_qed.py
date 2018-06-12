'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from LPGSOM import GSOM
from sklearn.datasets import make_blobs

X, c = make_blobs(3000, n_features=3, centers=2, cluster_std=1.3, random_state=10)#np.random.random((100, 3))



model = GSOM(lr=0.1, beta=0, sf = 0.2, fd=0.1, wd=0.02)

Y = model.fit_transform(X)
W = model.C

G = model.n_graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X.T[0], X.T[1], X.T[2], c = c)

for i in range(G.shape[0]):
    for j in range(i):
        if G[i, j]:
            ax.plot([W[i, 0], W[j, 0]], [W[i, 1], W[j, 1]],[W[i,2], W[j,2]], c='black')
print('Showing the 3d Plot')
# plt.show()
plt.show(block = False)

fig2 = plt.figure()
plt.scatter(Y.T[0], Y.T[1], c= c)
plt.show()