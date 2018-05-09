import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from MovingMap import  MovingMap
from SelfOrganizingSwarm import SelfOrganizingSwarm
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding, Isomap, SpectralEmbedding
from bgsom import GSOM as MLGSOM
from cgsom import GSOM as GSOM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize

''' Import Data from sklearn data sets'''

pack = datasets.load_digits()
D = normalize(pack.data)
c = pack.target

###### Test code for PCA ###########

print "################# PCA ###############"
amis = []
aris = []

for i in range(10):
    Reducer = PCA(2)
    Y = Reducer.fit_transform(D)
    labs = KMeans(10).fit(Y).labels_
    print '---- iteration ', str(i+1), ' --------'
    amis.append(adjusted_mutual_info_score(c, labs))
    aris.append(adjusted_rand_score(c, labs))

    print 'Adjusted Mutual Information :', amis[i]
    print 'Adjusted Rand Index         :', aris[i]

#----- average over the best 5 scores -------

ami_ = np.mean(np.array(amis)[np.array(amis).argsort()[5:]])
ari_ = np.mean(np.array(aris)[np.array(aris).argsort()[5:]])

print 'Average AMI : ', ami_
print 'Average ARI : ', ari_


###### Test code for LLE ###########

print "################# LLE ###############"

amis = []
aris = []

for i in range(10):
    Reducer = LocallyLinearEmbedding()
    Y = Reducer.fit_transform(D)
    labs = KMeans(10).fit(Y).labels_
    print '---- iteration ', str(i+1), ' --------'
    amis.append(adjusted_mutual_info_score(c, labs))
    aris.append(adjusted_rand_score(c, labs))

    print 'Adjusted Mutual Information :', amis[i]
    print 'Adjusted Rand Index         :', aris[i]

#----- average over the best 5 scores -------

ami_ = np.mean(np.array(amis)[np.array(amis).argsort()[5:]])
ari_ = np.mean(np.array(aris)[np.array(aris).argsort()[5:]])

print 'Average AMI : ', ami_
print 'Average ARI : ', ari_

###### Test code for MDS ###########


print "################# MDS ###############"

amis = []
aris = []

for i in range(10):
    Reducer = MDS()
    Y = Reducer.fit_transform(D)
    labs = KMeans(10).fit(Y).labels_
    print '---- iteration ', str(i+1), ' --------'
    amis.append(adjusted_mutual_info_score(c, labs))
    aris.append(adjusted_rand_score(c, labs))

    print 'Adjusted Mutual Information :', amis[i]
    print 'Adjusted Rand Index         :', aris[i]
ami_ = np.mean(np.array(amis)[np.array(amis).argsort()[5:]])
ari_ = np.mean(np.array(aris)[np.array(aris).argsort()[5:]])

print 'Average AMI : ', ami_
print 'Average ARI : ', ari_

print "################# Laplacian Eigenmaps ###############"

amis = []
aris = []

for i in range(10):
    Reducer = SpectralEmbedding()
    Y = Reducer.fit_transform(D)
    labs = KMeans(10).fit(Y).labels_
    print '---- iteration ', str(i+1), ' --------'
    amis.append(adjusted_mutual_info_score(c, labs))
    aris.append(adjusted_rand_score(c, labs))

    print 'Adjusted Mutual Information :', amis[i]
    print 'Adjusted Rand Index         :', aris[i]


#----- average over the best 5 scores -------

ami_ = np.mean(np.array(amis)[np.array(amis).argsort()[5:]])
ari_ = np.mean(np.array(aris)[np.array(aris).argsort()[5:]])

print 'Average AMI : ', ami_
print 'Average ARI : ', ari_

print "################# ISOMAP ###############"

amis = []
aris = []

for i in range(10):
    Reducer = Isomap()
    Y = Reducer.fit_transform(D)
    labs = KMeans(10).fit(Y).labels_
    print '---- iteration ', str(i+1), ' --------'
    amis.append(adjusted_mutual_info_score(c, labs))
    aris.append(adjusted_rand_score(c, labs))

    print 'Adjusted Mutual Information :', amis[i]
    print 'Adjusted Rand Index         :', aris[i]


#----- average over the best 5 scores -------

ami_ = np.mean(np.array(amis)[np.array(amis).argsort()[5:]])
ari_ = np.mean(np.array(aris)[np.array(aris).argsort()[5:]])

print 'Average AMI : ', ami_
print 'Average ARI : ', ari_