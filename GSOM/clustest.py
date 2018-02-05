import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from MovingMap import  MovingMap
from SelfOrganizingSwarm import SelfOrganizingSwarm
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
# from bgsom import GSOM
from cgsom import GSOM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize

pack = datasets.load_digits()
#datasets.load_iris()
D = normalize(pack.data)#[:1000]
# D/=D.max()
c = pack.target#[:1000]
Reducer = GSOM()#MovingMap(iterations=100, beta=0.5)


Y =Reducer.fit_transform(D, lr=1,  beta=0.3, sf=0.9, fd = 0.9, wd=0.025)#MDS().fit_transform(D)#
# Y =TSNE(perplexity=40).fit_transform(D)


labs = KMeans(10).fit(Y).labels_

print 'ami ', adjusted_mutual_info_score(c, labs)
print 'ars ', adjusted_rand_score(c, labs)

plt.subplot(211)
plt.scatter(Y.T[0], Y.T[1], s=15,alpha=0.35, edgecolors='none',  c = plt.cm.nipy_spectral(c.astype(float)/len(np.unique(c))))

classes = range(0, 10)

recs=[]
for i in range(0,10):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=plt.cm.nipy_spectral(i*1.0/10)))
# plt.legend(recs,classes,loc=4)
plt.subplot(212)
plt.scatter(Y.T[0], Y.T[1], s= 15, alpha=0.375, edgecolors='none',  c = 'gray')

plt.show()

