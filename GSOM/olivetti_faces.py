from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from HEXGSOM import GSOM

data = datasets.fetch_olivetti_faces()

X = data['images'].reshape(400, 64*64)

t = data['target']

model =GSOM(radius=6, min_rad=2, lrst=0.1,cluster_spacing_factor=.7, sd=0.0, labels=t)# TSNE()#

Y = model.fit_transform(X)

plt.scatter(Y.T[0], Y.T[1], c = t, cmap=plt.cm.jet, s = 15, alpha=0.5,edgecolors='none')
plt.show()

