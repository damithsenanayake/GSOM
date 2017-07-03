from AEGSOM import GSOM
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import KernelPCA,FactorAnalysis
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
rng = np.random.RandomState(seed=1234)
dat = rng.random_sample((100, 3))
from AutoEncoder import AutoEncoder


fi = pd.read_csv('../mnist_train.csv', header=None)
test = pd.read_csv('../mnist_test.csv', header=None)
dat =np.array(fi)[:, 1:]/255.0
labels = np.array(fi)[:, 0]
# reductor = AutoEncoder(hid_size=100)
# X = reductor.reduce(dat)

gsom = GSOM(dims=784, hid = 81, sf = 0.8, fd = 0.9, max_nodes = 2500, min_nodes = 10, radius=40, scale = 1, X=dat, nei=False, gaussian=True)


for i in range(1):
    print '\nbatch '+ str(i+1)
    prune = False
    if i %4 ==1:
        prune = True
    gsom.train_batch(dat[i*500:(i+1)*500], lr = 1*np.exp(-i/ 10),  iterations=50, prune=False)#lr=0.01*np.exp(-i/200), iterations=100)
    # if len(gsom.w) > gsom.max_nodes:
    #     gsom.cull_old()
# gsom.prune()
grid, hits = gsom.predict((dat[:500]))
x, y = grid.T
colors = dat
nodes = np.unique(np.array(hits))

batches = []

for n in nodes:
    ninds = np.where(np.array(hits)==n)[0]
    batches.append(dat[ninds])

plt.scatter(x, y, s=1, color='white')

for i, j, t in zip(x, y, labels):
    plt.text(i, j, t, color = plt.cm.Set1(t/10.0), fontsize = 12)
plt.show()

plt.scatter(x, y, s=1)


map = gsom.grid.values()
hits = gsom.hits.values()

x, y = np.array(map).T

for i, j, t in zip(x,y,hits):
    plt.text(i, j, t, color = 'green', fontsize = 12)

plt.show()

#
# nets = {}
# for n in range(nodes.shape[0]):
#     print 'training : ', nodes[n]
#     nets[nodes[n]] = AutoEncoder(vis=784, hid=100)
#     gsom.range = 5
#     neighborhood = gsom.get_neighbourhood(nodes[n])
#     for n in neighborhood:
#         lr = gsom.grid[n] - gsom.
#     nets[nodes[n]].train(batches[n]/255.0, 10000, 0.1)#,batch_size=
#                  #2)
#
# predictions = []
#
# for net in nets.keys():
#     pred_n = nets[net].predict(dat[:1000]/255.0)
#     err = np.linalg.norm(pred_n - dat[:1000]/255.0, axis=1)
#     predictions.append(err)
#
# errors = np.array(predictions)
#
# x =[]
# y = []
#
# for i in range(errors.shape[1]):
#     node = gsom.grid[nodes[np.argmin(errors[:, i])]]
#     x.append(node[0])
#     y.append(node[1])
#
#
# plt.scatter(x, y, s=1)
#
# for i, j, t in zip(x, y, labels):
#     plt.text(i, j, t, color = 'purple', fontsize = 12)
# plt.show()