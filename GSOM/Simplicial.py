import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs,make_swiss_roll, make_s_curve
from sklearn.decomposition import PCA
from GAPENGSOM import GSOM
from sklearn.manifold import TSNE
# centers = np.array([[2, 0, 0], [-1, np.sqrt(3), 0], [-1, -np.sqrt(3),0], [0,0, np.sqrt(8)], [0,0, np.sqrt(8)/3]])
centers = np.array([[1,0,0],[-1, 0, 0],[0, 1, 0],[0, -1, 0], [0, 0, 0], [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1]])
randnoise = np.random

X=[]
c = []
i = 1
for cent in centers:
    x = np.random.randn(500, 3)#make_blobs(n_samples=1000, n_features=3, centers=1, cluster_std=0.5)
    t = np.ones(500)
    x -= x.min()
    # x /= 2
    x /= x.max()
    x /=0.5
    # x *=2
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
X +=10000
# reds = np.zeros((X.shape[0], 100))
#
# X = np.concatenate((X, reds), axis=1)
c = np.array(c).flatten(order=1)
# X, c = make_s_curve(2500)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(X.T[0], X.T[1], X.T[2], c=c, cmap=plt.cm.gist_rainbow)

plt.show(block=False)
print np.linalg.norm(X - X[1], axis = 1)
model = GSOM(lrst=.1,  sf_min=0.9, sf_max=0.9, fd = .2, radius=4, min_rad =2)#UMAP()#

Y = model.fit_transform(X)#PCA().fit_transform(X)
fig = plt.figure()
# ax = fig.add_subplot(212)
plt.scatter(Y.T[0], Y.T[1], c= c, cmap=plt.cm.gist_rainbow, alpha=0.4, s=20)
plt.show()
