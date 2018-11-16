import matplotlib.pyplot as plt
import  pandas as pd
import timeit
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import pairwise_distances
from HEXGSOM import GSOM

from matplotlib.collections import  LineCollection

import gc

model = GSOM()

print GSOM.__module__
fi = pd.read_csv('~/data/fashionmnist/fashion-mnist_train.csv')
# test = pd.read_csv('../mnist_test.csv', header=None)
samples = 2000

dat =(np.array(fi)[:samples, 1:]).astype(float)#/255.0
order = np.random.permutation(range(samples))
dat = PCA(15, random_state=1).fit_transform(dat)[order]
dat -= dat.min()
dat /= dat.max()
labels = np.array(fi)[:samples, 0].astype(int)[order]

# dat = dat[(labels ==5) | (labels == 3)]
# labels = labels[(labels == 5)|( labels == 3)]
del fi
print dat.shape

gc.collect()
# x, y = SelfOrganizingSwarm(iterations=10, alpha=1, beta=0.1, delta=0, theta=3.5).fit_transform(dat[:samples]).T
'''Cluster Separation Factor :
    dix = 1/(1-csf)
'''

st = timeit.default_timer()
model = GSOM(lrst=.1, sf=0.9, fd = .99, radius=4, min_rad = 2., sd=.08, its=10, cluster_spacing_factor=.8, labels = labels, momentum=0.8)
# model = TSNE(perplexity=40)#
# x, y = MovingMap(iterations=100, beta=1.5).fit_transform(dat[:samples]).T
Y= model.fit_transform(dat)

graph = np.zeros((model.W.shape[0], model.W.shape[0]))

pw_grid = pairwise_distances(model.grid, model.grid)

pw_weights = pairwise_distances(model.W, model.W)

graph[pw_grid-1<=1.e-5]=1

pw_weights *= graph
pw_weights/=pw_weights.max()
fig = plt.figure(figsize=(5, 10))
plt.subplot(211)

x, y = Y.T
plt.scatter(x, y, edgecolors='none',c=plt.cm.jet(labels/10.), alpha = 0.5, s = 15, marker='h')
plt.subplot(212)
for i in range(graph.shape[0]):
    for j in range(graph.shape[0]):
        if graph[i,j]:

            plt.plot([model.grid[i,0], model.grid[j,0]], [model.grid[i,1], model.grid[j,1]], c='b', linewidth = 2*pw_weights[i,j])


plt.show()
