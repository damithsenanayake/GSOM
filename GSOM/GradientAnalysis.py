from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import  KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ars
from sklearn.preprocessing import normalize
from cgsom import GSOM as GSOM
from RandomSampleGSOM import GSOM as DSGSOM

n_clusters = 10

rand = np.random.RandomState(1)
X, color = datasets.samples_generator.make_blobs(n_samples=3000, n_features=50, centers=n_clusters, cluster_std=3.5, random_state=20)

n_fac = .3
s_f = np.sqrt(X.var(axis=0))
noise = rand.random_sample(X.shape)
noise *= 2*s_f
noise -= s_f
noise /= n_fac
X += noise

configs = []

configs.append(GSOM())
configs.append(DSGSOM())

X=  normalize(X)

# fig = plt.figure()
model = DSGSOM( lr = .1, beta=0.0, sf=0.8, wd=0.375, fd=0.9)
Y= model.fit_transform(X)#X,lr = 1.0, beta=0.0,sf=0.01, fd=0.75, wd=0.5)
distas = model.distas
deltas = model.deltas
tx =[]
ty = []
lens = []
for i in distas.keys():
    tx.append(np.mean(deltas[i]))
    ty.append(np.mean(distas[i]))
    lens.append(len(distas[i]))

order = np.argsort(ty)
plt.scatter(np.array(ty)[order], np.array(tx)[order], alpha=0.5, s = 5, c = lens, cmap=plt.cm.jet)
# plt.plot(np.array(ty)[order], np.array(tx)[order], c = 'grey')
plt.colorbar()
plt.show()
# fig = plt.figure()
# ax = Axes3D(fig)00
# ax.scatter(X.T[0], X.T.[1], X.T[2],c = color, alpha=0.5, edgecolors='none')
# plt.show()
# plt.subplot(211)
# ax = fig.add_subplot(211)
fig = plt.figure(num=None, figsize=(8,8))
plt.scatter(Y.T[0], Y.T[1], s = 6, c = plt.cm.jet(color/(n_clusters*1.0)), edgecolors='none', alpha=0.375)


labs = KMeans(n_clusters).fit(Y).labels_

# plt.subplot(212)
# plt.scatter(Y.T[0], Y.T[1], s = 15, c =plt.cm.jet(labs/(n_clusters*1.0)), edgecolors='none', alpha=0.375)


print 'ars ', ars(color,labs)
print 'ami ', ami(color, labs)


#
# Y = Isomap().fit_transform(X)
# ax2 = fig.add_subplot(121)
# ax2.scatter(Y.T[0], Y.T[1], c = color, edgecolors='none', alpha=0.5)

plt.show()