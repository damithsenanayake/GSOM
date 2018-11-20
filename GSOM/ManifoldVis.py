import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_s_curve

from SFGSOM import GSOM

X1 = np.random.random((1000, 3))
X2 = np.random.random((1000, 3))

X1+= 2


X = np.concatenate((X1, X2), axis=0)
c = X/X.max()
# X, c = make_s_curve(n_samples=15000)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(X.T[0], X.T[1], X.T[2], c=c)
model = GSOM(radius=8, min_rad=2,sf_max=0.9, sf_min=0.9)
Y = model.fit_transform(X)

grid = model.grid
W = model.W
G = np.zeros((grid.shape[0], grid.shape[0]))
pds = pairwise_distances(grid, grid)
for i in range(grid.shape[0]):
    neis = np.where(pds[i]==1)[0]
    G[i][neis]=1

for i in range(G.shape[0]):
    for j in range(i):
        if G[i, j]:
            ax.plot([W[i, 0], W[j, 0]], [W[i, 1], W[j, 1]],[W[i,2], W[j,2]], c='black')
plt.show(block=False)

fig = plt.figure()
plt.scatter(Y.T[0], Y.T[1], c = c)
plt.show()