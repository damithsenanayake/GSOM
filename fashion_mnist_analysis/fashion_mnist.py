import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
from geg_som import GEGSOM
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.decomposition import PCA
import timeit

from umap import UMAP

fi = pd.read_csv('~/data/fashionmnist/fashion-mnist_train.csv')
samples = 6000

dat =(np.array(fi)[:samples, 1:]).astype(float)
order = np.random.permutation(range(samples))
dat = PCA(18, random_state=1).fit_transform(dat)
dat -= dat.min()
dat /= dat.max()
labels = np.array(fi)[:samples, 0].astype(int)

print dat.shape

gc.collect()

st = timeit.default_timer()
model = GEGSOM(labels=labels, verbose=True)#Provide labels to store intermediate results

Y= model.fit_transform(dat)

et = timeit.default_timer() - st
hrs = np.floor(et/3600)
mins = np.floor((et - 3600*hrs)/60)
secs = et-3600*hrs - 60* mins
print 'Total time : ', hrs, ':', mins, ':', secs

x, y = Y.T

clusterer = KMeans(10)
kl = clusterer.fit(Y).labels_
print 'instances : ', samples
print 'ars :', adjusted_rand_score(labels, kl)
print 'ami :', adjusted_mutual_info_score(labels, kl)
fig = plt.figure(figsize=(5, 10))
plt.subplot(211)

np.savetxt('mnist_'+str(samples)+'.csv', np.concatenate((Y, np.array([labels]).T),axis=1))
plt.scatter(x, y, edgecolors='none',c=plt.cm.jet(labels/10.), alpha = 0.5, s = 15, marker='h')
plt.subplot(212)
plt.scatter(x, y, edgecolors='none',c=plt.cm.jet(kl/10.), alpha = 0.5, s = 15, marker='h')


plt.show()
