import numpy as np
import scipy.stats as st
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

def sigmoid(x):
    return (1./(1+np.exp(-x)))

# x1 = np.array(range(100)).astype(float)#np.random.randn(100).astype(float)
# x1/=x1.max()*.2
# x = np.array(range(100)).astype(float)#np.random.random(1000)
# x -= x.min()
# #
# x/=x.max()
# x.sort()
d = 30

X = np.random.randn(100, d)
x =np.array(range(100)).astype(float)
x/=x.max()
p = 1.2
D = cdist(np.zeros((1, d)), X, metric='minkowski', p = p)#**p
D.sort()
# y =np.exp(-x1**2/np.mean(2*x1**2))
y = D#1-np.exp(-(32./1155)**2*(x1)**8)
# y1 = 1-np.exp(-.2 * (x1)**8)
y2 = 1-np.exp(-1000 * (x)**4)

plt.scatter(x, y2, c = plt.cm.jet(x/x.max()))
# # plt.scatter(x1, y1, c = plt.cm.jet(x1))
# plt.scatter(x*x1.max(), y2, c = plt.cm.jet(x))

plt.show()
