import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

def sigmoid(x):
    return (1./(1+np.exp(-x)))

x = np.array(range(100))
y =np.exp(-50.5*(x/float(np.mean(x)))**(60))#(1.+(x/100.)**2)**-6#sigmoid(x**2/100.**2)#
plt.plot(x, 1-y)
plt.show()
