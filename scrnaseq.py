import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from GSOM.tgsom import GSOM


tb = np.loadtxt('/home/senanayaked/data/GSE/table.txt',delimiter='\t', dtype='str')

orig = tb[5:, 1:].astype(float).T

gbc = normalize(orig)

vars = gbc.var(axis=0)

means = gbc.mean(axis=0)
print vars.shape

selected= len(np.where(np.sqrt(np.sqrt(vars))>0.12)[0])

selargs = np.argsort(np.sqrt(np.sqrt(vars)))[-selected:]

X = orig.T[selargs].T

x, y = GSOM().fit_transform(normalize(X), lr = 1.0, beta=0.2, sf=0.999, wd=0.0275, fd=1.8).T
# x, y = TSNE().fit_transform(normalize(X)).T

colors = plt.cm.rainbow(np.array(range(x.shape[0])).astype(float)/x.shape[0])

plt.scatter(x,y, c=colors, alpha = 0.6)
plt.show()