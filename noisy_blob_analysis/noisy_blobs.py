import numpy as np
from sklearn.datasets import make_blobs
from geg_som import GEGSOM
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from umap import UMAP



configurations = [#[9, 10, 1],
    #               [9, 10, 4],
    # [9, 10, 5],
    #
    # [9, 10, 6],
                  [9, 10, 8],
                  [9, 30, 1],
                  [9, 30, 4],
                    [9, 30, 5],

                    [9, 30, 6],

                    [9, 30, 8]]


for config in configurations:
    print config

    X, c = make_blobs(6000, n_features=config[0], centers=config[1], cluster_std=config[2], random_state=1, )

    values = []
    order = np.random.permutation(range(X.shape[0]))
    X = X[order]
    c = c[order]

    models = [
        # UMAP(),
        GEGSOM(labels=c),
        # TSNE(perplexity=40)
    ]

    scores = []

    for model in models:
        print str(model.__class__).split(".")[-1], '            ',

    print ''

    for model in models:

        Y = model.fit_transform(X)#PCA().fit_transform(X)

        clusterer = KMeans(np.unique(c).shape[0])
        clusterer.fit(Y)
        preds = clusterer.labels_

        ars = adjusted_rand_score(c, preds)
        ami = adjusted_mutual_info_score(c, preds)

        print  str(ars), ' / ', str(ami),'  ',
        scores.append(str(model.__class__)+"\t"+str(ars)+"\t"+str(ami))
    print ''
    values.append(str(config))
    values.append(scores)

print values

