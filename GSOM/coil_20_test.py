import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from HEXGSOM import  GSOM
# from protodsgsom import GSOM
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
# print "** PROTODGSOM **"
fread = np.array(pd.read_csv("/home/senanayaked/data/coil20.csv", header=None))
print GSOM.__module__

X = fread[:, :-1]
# X = X/X.max()
X = PCA(30).fit_transform(X)
t = fread[:,-1]
imlables = ['duck', 'block 1', 'car 1', 'fs cat', 'Anc', 'Car 2', 'block 2', 'baby powder', 'Tylenol', 'Vaseline', 'Mushroom', 'cup', 'piggy', 'socket', 'pot', 'bottle', 'dish', 'cup 2', 'car 3', 'tub']
model = GSOM(lrst=.5,sf=0.9, fd = .9, radius=6, min_rad =6., sd=0.08, cluster_spacing_factor=.85,momentum=0., its=100, labels=t, neighbor_func='gaussian', map_structure=8)#TSNE(perplexity=40)#
Y=model.fit_transform(X)# GSOM().fit_transform(X, lr = 1.0, beta=0.5, sf=0.995, wd=0.02, fd=1.9)


unique = list(set(t))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
for i, u in enumerate(unique):
    xi = [Y.T[0][j] for j  in range(len(Y.T[0])) if t[j] == u]
    yi = [Y.T[1][j] for j  in range(len(Y.T[1])) if t[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(imlables[int(u)-1]), alpha=0.7, edgecolors='none')
# plt.legend()

plt.show()
