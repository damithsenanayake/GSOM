import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SelfOrganizingSwarm import SelfOrganizingSwarm as SOS

from sklearn.cluster import AffinityPropagation

from sklearn.manifold import TSNE


dataset = pd.read_csv('~/data/gene_expr_170104.csv')
data = np.array(dataset)[:, 1:].astype(float).T




Y = TSNE().fit_transform(data)
clus = AffinityPropagation()

lab = clus.fit_predict(Y)

x, y  = Y.T



plt.scatter(x, y, alpha=0.9, c = plt.cm.Spectral(lab.astype(float) / lab.max()), edgecolors='none')
# for i, j, t in zip(x, y, range(x.shape[0])):
#     plt.text(i, j, t, color = 'purple')

plt.show()

x, y, = SOS(iterations=10, alpha=1, beta=0, delta=0, theta=3.5).fit_transform(data).T

plt.scatter(x, y, alpha=0.4, c = plt.cm.Spectral(lab.astype(float) / lab.max()), edgecolors='none')
# for i, j, t in zip(x, y, range(x.shape[0])):
#     plt.text(i, j, t, color= 'purple' )

plt.show()