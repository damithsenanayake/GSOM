import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

def sigmoid(x):
    return (1./(1+np.exp(-x)))

x = np.array(range(100))
y = np.exp(-1.7*(x/100.)**(5))#sigmoid(x**2/100.**2)#
plt.plot(x, 1-y)
plt.show()
