import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SelfOrganizingSwarm import SelfOrganizingSwarm
from MovingMap import MovingMap
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import PCA
X = np.array(pd.read_csv('~/data/spiral.csv', header=None))#np.random.random((100, 3))
X = np.append(X, np.random.random((X.shape[0], 100)), axis=1)
s = MovingMap()#(iterations=1000, alpha=1, beta=0.5, delta=0.0)
s.fit(X[np.random.permutation(X.shape[0])])
Y = s.predict(X)

colors = np.array(['blue', 'green', 'orange', 'red'])
plt.scatter(Y.T[0], Y.T[1], s=75, c =plt.cm.Set1(np.array(range(X.shape[0]))*1.0/X.shape[0]), edgecolors='none', alpha=0.25)
# plt.plot(Y.T[0], Y.T[1], c='grey')
plt.show()