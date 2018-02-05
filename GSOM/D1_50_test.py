import pandas as pd
import numpy as np
from WGSOM import GSOM
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA

data = pd.read_csv('~/data/D1_50.csv', header=None)

X= np.array(data)[:634, :6]
print X.shape
Y = np.array(data)[:634, -1].astype(int)
X_ = (X+abs(X.min()))
X_/=X_.max()

gsom = GSOM(dims=6,  hid=2, fd=0.8, sf=0.75, radius=25, max_nodes=4000, min_nodes=10, gaussian=True, nei=True, X=X_)


for i in range(200):

    print "\n iteration ", (i+1)
    gsom.train_batch(X_, 1, 1*np.exp(-i/40), i<100)

grid, hits = gsom.predict(X_)
x, y = grid.T
# x, y = TSNE(2).fit_transform(X_[:, :15]).T


colors = ["", "green", "blue", "red"]

plt.scatter(x, y, edgecolors='none', c = np.array(colors)[Y] , alpha = 0.5)
plt.show()

plt.scatter(x[:191], y[:191], edgecolors='none', c = np.array(colors)[Y[:191]])
plt.show()

plt.scatter(x[191:430], y[191:430], edgecolors='none',  c = np.array(colors)[Y[191:430]])
plt.show()

plt.scatter(x[430:], y[430:], edgecolors='none', c=np.array(colors)[Y[430:]])
plt.show()