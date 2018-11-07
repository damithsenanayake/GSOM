import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs,make_swiss_roll, make_s_curve
from sklearn.decomposition import PCA
from HEXGSOM import GSOM
from sklearn.manifold import TSNE
# centers = np.array([[2, 0, 0], [-1, np.sqrt(3), 0], [-1, -np.sqrt(3),0], [0,0, np.sqrt(8)], [0,0, np.sqrt(8)/3]])
centers = np.array([[1,0,0],[-1, 0, 0],[0, 1, 0],[0, -1, 0], [0, 0, 0], [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1]])
randnoise = np.random
print GSOM.__module__
X=[]
c = []
i = 1
for cent in centers:
    x = np.random.randn(1000, 3)#make_blobs(n_samples=1000, n_features=3, centers=1, cluster_std=0.5)
    t = np.ones(1000)
    x -= x.min()
    # x /= 2
    x /= x.max()
    x -=0.5
    x *=2
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

# X = np.array(X)
# X +=10000
# reds = np.zeros((X.shape[0], 100))
#
# X = np.concatenate((X, reds), axis=1)
# X, c = make_blobs(6000, n_features=9, centers=10, cluster_std=8, random_state=1, )
# c = np.array(c).flatten(order=1)
# X, c = make_s_curve(2500)

order = np.random.permutation(range(X.shape[0]))
X = X[order]
c = c[order]
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
P = PCA(3).fit_transform(X)
ax.scatter(P.T[0], P.T[1], P.T[2], c=c, cmap=plt.cm.jet)

plt.show(block=False)
# print np.linalg.norm(X - X[1], axis = 1)
model = GSOM(lrst=.05, sf=0.8, fd = .1, radius=4, min_rad =3, sd=0.04, its= 15, cluster_spacing_factor=0.8, labels=c)#UMAP()#
# model = TSNE(perplexity=40)

Y = model.fit_transform(X)#PCA().fit_transform(X)
fig = plt.figure()
# ax = fig.add_subplot(212)
plt.scatter(Y.T[0], Y.T[1], c= c, cmap=plt.cm.jet, alpha=0.4, s=15, edgecolors='none')
plt.show()
