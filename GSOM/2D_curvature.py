import numpy as np
import matplotlib.pyplot as plt
from TGSOM import GSOM
centers = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 0]])

X = np.zeros((1000, 2))
cs = np.zeros(1000)
i = 0
for c in centers:

    x = np.random.randn(200, 2)
    x -= x.min()
    x /= x.max()
    x -=0.5
    X[i*200:(i+1)*200] = x+c
    cs[i*200:(i+1)*200] = i
    i+= 1

fig = plt.figure()

Y = GSOM(lrst=.1, radius=6).fit_transform(X)
plt.subplot(211)
plt.scatter(Y.T[0], Y.T[1], c= cs, cmap=plt.cm.gist_rainbow, alpha = 0.8)
plt.subplot(212)
plt.scatter(X.T[0], X.T[1], c= cs, cmap=plt.cm.gist_rainbow, alpha = 0.8)

plt.show()
