import numpy as np
import matplotlib.pyplot as plt
from AEGSOM import  GSOM
import pandas as pd
from sgsom import GSOM as GS

rgen = np.random.RandomState(1234)

X = np.array(pd.read_csv('~/data/4C_2M.csv', header=None))
# gsom = GSOM(dims=3, hid = 1,  fd=0.1, sf=0.075, radius=5,max_nodes=1000, min_nodes=100, X = X,nei=True,gaussian=True)
gsom = GS(dims=2, sf=0.7, fd=0.25, max_nodes=10000, min_nodes=5, radius=5)
for i in range(1):
    print "\n iteration ", (i+1)
    gsom.train_batch(X, 20, 1, False)


gsom.smooth_batch(X, 20, 0.01, False)

grid, hits = gsom.predict(X)
x, y = grid.T
plt.scatter(x, y, alpha = 0.75, c= np.array(range(X.shape[0]))/250, edgecolors='none' , s=50)
plt.show()

x, y = np.array(gsom.grid.values()).T

hits = np.array(gsom.hits.values())


# plt.scatter(x, y, edgecolors='none', c='white')
# for i, j, t in zip(x,y,hits):
#     plt.text(i, j, t, color = plt.cm.Set1(t*1.0/np.max(hits)), fontsize = 12)
#
# plt.show()
#
