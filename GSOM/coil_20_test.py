import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from bgsom import  GSOM
from GASMAP import GASMAP
# from protodsgsom import GSOM
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
# print "** PROTODGSOM **"
fread = np.array(pd.read_csv("/home/senanayaked/data/coil20_filtered.csv", header=None))
print GSOM.__module__

X = fread[:, :-1]
# X = X/X.max()
t = fread[:,-1]
imlables = ['duck', 'block 1', 'car 1', 'fs cat', 'Anc', 'Car 2', 'block 2', 'baby powder', 'Tylenol', 'Vaseline', 'Mushroom', 'cup', 'piggy', 'socket', 'pot', 'bottle', 'dish', 'cup 2', 'car 3', 'tub']

# Y= GSOM().fit_transform(X,lr = 1.0, beta=0.99, sf=0.99, wd=0.012, fd=1.9)

Y = GASMAP().fit_transform(X, t)

unique = list(set(t))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
for i, u in enumerate(unique):
    xi = [Y.T[0][j] for j  in range(len(Y.T[0])) if t[j] == u]
    yi = [Y.T[1][j] for j  in range(len(Y.T[1])) if t[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(imlables[int(u)-1]), alpha=0.7, edgecolors='none')
plt.legend()

plt.show()
