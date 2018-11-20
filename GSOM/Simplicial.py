import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs,make_swiss_roll, make_s_curve
from sklearn.decomposition import PCA
from HEXGSOM import GSOM
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from umap import UMAP

# centers = np.array([[2, 0, 0], [-1, np.sqrt(3), 0], [-1, -np.sqrt(3),0], [0,0, np.sqrt(8)], [0,0, np.sqrt(8)/3]])
centers = np.array([[1,0,0],[-1, 0, 0],[0, 1, 0],[0, -1, 0], [0, 0, 0], [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1]])
randnoise = np.random
print GSOM.__module__
X=[]
c = []
i = 1
for cent in centers:
    x = np.random.randn(500, 3)
    t = np.ones(500)
    x -= x.min()
    # x /= 2
    x /= x.max()
    x -=0.5
    x *=2
    x += cent
    try:
        X = np.concatenate((X,x), axis=0)
    except:
        X = x
    try:
        c = np.concatenate((c, t+i))
    except:
        c = t+i
    i+= 1


configurations = [[9, 10, 1],
                  [9, 10, 4],
                  [9, 10, 8],
                  [9, 20, 1],
                  [9, 20, 4],
                  [9, 20, 8]]

for config in configurations:
    print config

    X, c = make_blobs(6000, n_features=config[0], centers=config[1], cluster_std=config[2], random_state=1, )

    values = []
    order = np.random.permutation(range(X.shape[0]))
    X = X[order]
    c = c[order]

    models = [
        UMAP(),
        GSOM(lrst=.5, sf=0.9, fd=.9, radius=8, min_rad=2, sd=0.02, its=20, cluster_spacing_factor=1., labels=c,
             momentum=.0, neighbor_func='cut_gaussian'),
        TSNE(perplexity=40)
    ]

    scores = []
    for model in models:
        print '!-------------- ', str(model.__class__), ' -----------------!'

        Y = model.fit_transform(X)#PCA().fit_transform(X)

        clusterer = KMeans(np.unique(c).shape[0])
        clusterer.fit(Y)
        preds = clusterer.labels_

        ars = adjusted_rand_score(c, preds)
        ami = adjusted_mutual_info_score(c, preds)

        print 'ars : ', str(ars)
        print 'ami : ', str(ami)
        scores.append(str(model.__class__)+"\t"+str(ars)+"\t"+str(ami))
    values.append(str(config))
    values.append(scores)

print values


# fig = plt.figure()
# # ax = fig.add_subplot(212)
# plt.scatter(Y.T[0], Y.T[1], c= c, cmap=plt.cm.jet, alpha=0.4, s=15, edgecolors='none')
# plt.show()
