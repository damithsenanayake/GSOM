import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from
centers = np.array([[2, 0, 0], [-1, np.sqrt(3), 0], [-1, -np.sqrt(3),0], [0,0, np.sqrt(8)]])

randnoise = np.random

X=[]
c = []
i = 1
for cent in centers:
    x = np.random.randn(500, 3)#make_blobs(n_samples=1000, n_features=3, centers=1, cluster_std=0.5)
    t = np.ones(500)
    x -= x.min()
    x /= 2
    # x /= x.max()

    x += cent
    try:
        X = np.concatenate((X,x), axis=0)#X.append(x)
    except:
        X = x
    try:
        c = np.concatenate((c, t+i))#append(t+i)
    except:
        c = t+i
    i+= 1

X = np.array(X)
c = np.array(c).flatten(order=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(X.T[0], X.T[1], X.T[2], c=c)

plt.show(block=False)
print np.linalg.norm(X - X[1], axis = 1)

Y = PCA().fit_transform(X)
fig = plt.figure()
# ax = fig.add_subplot(212)
plt.scatter(Y.T[0], Y.T[1], c= c, alpha=0.4)
plt.show()