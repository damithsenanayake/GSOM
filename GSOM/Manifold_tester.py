import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as ds
from sklearn.manifold import TSNE, MDS
from growthdsgsom import GSOM
print GSOM.__module__

X, t = ds.make_swiss_roll(1500, random_state=20)

Z, t2 = ds.make_s_curve(1500, random_state=1)

rand = np.random.RandomState(seed=10)
Z += rand.randn(3)*5

X = np.concatenate((X/20, Z/20), axis=0)

gsom = GSOM(lr = 0.5, beta=0.1, sf=0.9999, wd=0.0035, fd=1.9)

Y = gsom.fit_transform(X)

'''uncomment following line to run for t-SNE. Comment above line'''

# Y = TSNE().fit_transform(X)

Yc = Y[:1500].T
Ym = Y[1500:].T

plt.scatter(Yc[0], Yc[1], c = t, cmap = plt.cm.Spectral, alpha = 0.5)
plt.scatter(Ym[0], Ym[1], c = t2/2.0 , cmap=plt.cm.rainbow, alpha = 0.5)
plt.show()