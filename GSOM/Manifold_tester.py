import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as ds
from sklearn.manifold import TSNE, MDS
from dsgsom import GSOM
print GSOM.__module__

X, t = ds.make_swiss_roll(50000, random_state=20)

# Z, t2 = ds.make_s_curve(1500, random_state=1)
#
# rand = np.random.RandomState(seed=10)
# Z += rand.randn(3)*10
#
# X = np.concatenate((X/20, Z/20), axis=0)

gsom = GSOM(lr = 0.25, beta=0., sf=0.9, wd=0.00, fd=.9)

Y = gsom.fit_transform(X)

'''uncomment following line to run for t-SNE. Comment above line'''

# Y = TSNE().fit_transform(X)

Yc = Y.T#[:1500].T
# Ym = Y[1500:].T

plt.scatter(Yc[0], Yc[1], c = t, cmap = plt.cm.Spectral, alpha = 0.5)
# plt.scatter(Ym[0], Ym[1], c = t2/2.0 , cmap=plt.cm.rainbow, alpha = 0.5)
plt.show()
