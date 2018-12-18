import numpy as np
import scipy.stats as st
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def sigmoid(x):
    return (1./(1+np.exp(-x)))


def _curse_curve(x, a):
    return a*np.log(x)


# x1 = np.array(range(100)).astype(float)#np.random.randn(100).astype(float)
# x1/=x1.max()*.2
# x = np.array(range(100)).astype(float)#np.random.random(1000)
# x -= x.min()
# #
# x/=x.max()
# x.sort()
d = 30

X = np.array(range(100)).astype(float)#100, d)
X/=X.max()
x =np.array(range(100)).astype(float)
x/=x.max()
p = 2
# D = cdist(np.zeros((1, d)), X, metric='minkowski', p = p)#**p
# D.sort()
# D/=D.max()
k = 7
a = 2
b = 6
# y =np.exp(-x1**2/np.mean(2*x1**2))
y = 1+50*np.exp(-a*(X)**b)#1-np.exp(-(32./1155)**2*(x1)**8)
# y1 = 1-np.exp(-.2 * (x1)**8)
# y2 = 1-np.exp(-1000 * (x)**4)

plt.scatter(x, y, c = plt.cm.jet(x/x.max()))
# # plt.scatter(x1, y1, c = plt.cm.jet(x1))
# plt.scatter(x*x1.max(), y2, c = plt.cm.jet(x))

plt.show()
