import numpy as np
import matplotlib.pyplot as plt
from RandomBatchGSOM import GSOM
from sklearn.manifold import TSNE
centers = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 0]])

c_size = 500
n = c_size*5
X = np.zeros((n, 2))
cs = np.zeros(n)
i = 0
for c in centers:

    x = np.random.randn(c_size, 2)
    x -= x.min()
    x /= x.max()
    x -=0.5
    X[i*c_size:(i+1)*c_size] = x+c
    cs[i*c_size:(i+1)*c_size] = i
    i+= 1

fig = plt.figure()

Y = GSOM(lrst=.1, radius=10).fit_transform(X)
plt.subplot(311)
plt.scatter(Y.T[0], Y.T[1], c= cs, cmap=plt.cm.gist_rainbow, alpha = 0.8)
plt.subplot(312)

Y = TSNE().fit_transform(X)

plt.scatter(Y.T[0], Y.T[1], c= cs, cmap=plt.cm.gist_rainbow, alpha = 0.8)
plt.subplot(313)
plt.scatter(X.T[0], X.T[1], c= cs, cmap=plt.cm.gist_rainbow, alpha = 0.8)

plt.show()
