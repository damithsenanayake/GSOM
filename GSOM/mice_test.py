import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
from bgsom import GSOM
from cgsom import GSOM as GS
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score


fread = pd.read_excel('~/data/mice/Data_Cortex_Nuclear.xls').fillna(0)

classes = np.array(fread)[:, -1]
cls = LabelEncoder().fit_transform(classes)
c=cls.astype(float)/cls.max()


X = normalize(np.array(fread)[:, 1:-4].astype(float))
print X.shape
Y = GSOM().fit_transform(X, lr=1,  beta=0.08, sf=0.997, fd = 1.99, wd=0.0079)
# Y=TSNE(perplexity=60).fit_transform(X)
# Y = GS().fit_transform(X, lr=1,  beta=0.07, sf=0.995, fd = 0.99, wd=0.00)
# Y = PCA(2).fit_transform(X)

# Y = LocallyLinearEmbedding(100).fit_transform(X)


preds = KMeans(8).fit(Y).labels_

T = cls

ami = adjusted_mutual_info_score(T, preds)
ars = adjusted_rand_score(T, preds)

print 'ami : ', str(ami)
print 'ars : ', str(ars)

plt.scatter(Y.T[0], Y.T[1], c=plt.cm.nipy_spectral(c), alpha=0.5)
recs=[]
for i in range(0,8):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=plt.cm.nipy_spectral(i*1.0/7)))
# plt.legend(recs,np.unique(classes),loc=4)

plt.show()