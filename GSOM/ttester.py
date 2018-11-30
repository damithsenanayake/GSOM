import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

def sigmoid(x):
    return (1./(1+np.exp(-x)))

x1 = np.array(range(100)).astype(float)#np.random.randn(100).astype(float)
x1/=x1.max()
x = np.random.random(100)
x -= x.min()

x/=x.max()
x.sort()



y =np.exp(-x**2/np.mean(2*x**2))#np.exp(-(10*(1-.99))*x**60)
z = 1./(1+3.5*x**2)

plt.scatter(x, y, c = plt.cm.jet(x))

plt.show()
