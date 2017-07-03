import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AEGSOM import GSOM
zoo = pd.read_csv('~/data/zoo.data.txt', header=None)
zoo = np.array(zoo)
names = zoo[:, 0]

classes = zoo[:, -1].astype(float)
X = zoo[:, 1:-1].astype(float)
X/= X.max()

gsom = GSOM(dims=X.shape[1], hid=3, radius=5, fd=0.8, sf=0.9, nei=True, gaussian= True, min_nodes=10, max_nodes=250, X =X)

for i in range(100):

    print "\n iteration ", (i+1)
    gsom.train_batch(X, 1, 1*np.exp(-i/10), False)


grid, hits = gsom.predict(X)

x, y = grid.T

plt.scatter(x, y, marker = 3, edgecolors='none', c = 'white' )

for i, j, t, c in zip(x, y, names, classes):
    plt.text(i+np.random.random(1)*0.5, j+np.random.random(1)*0.5, t, color=plt.cm.Set1(c/10.0), fontsize=10)

plt.show()