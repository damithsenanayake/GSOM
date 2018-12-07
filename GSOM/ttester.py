import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

def sigmoid(x):
    return (1./(1+np.exp(-x)))

x1 = np.array(range(100)).astype(float)#np.random.randn(100).astype(float)
x1/=x1.max()*.5
# x = np.random.random(1000)
# x -= x.min()
#
# x/=x.max()
# x.sort()



# y =np.exp(-x1**2/np.mean(2*x1**2))
y = 1-np.exp(-5*(x1)**8)
y1 = 1-np.exp(-.2 * (x1)**8)
y2 = np.exp(-40 * (1-x1)**2)

plt.scatter(x1, y, c = plt.cm.jet(x1))
plt.scatter(x1, y1, c = plt.cm.jet(x1))
plt.scatter(x1, y2, c = plt.cm.jet(x1))

plt.show()
