import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from bgsom import  GSOM
import sklearn.datasets as ds
from sklearn.preprocessing import normalize

data = ds.fetch_olivetti_faces()

X = data.images.reshape((len(data.images), -1))

t = data.target


print X.shape

Y = GSOM().fit_transform(X,lr = 1.0, beta=0.4, sf=0.999, wd=0.005, fd=1.9999)#
# Y =  TSNE().fit_transform(X)#

plt.scatter(Y.T[0], Y.T[1], c = t, cmap = plt.cm.Set1)
plt.show()