import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

def sigmoid(x):
    return (1./(1+np.exp(-x)))

x = np.array(range(100))
y =np.exp(-4.5*((x)/float(np.amax(x)))**(4))#(1.+(x/100.)**2)**-6#sigmoid(x**2/100.**2)#
plt.plot(x,y)
plt.show()
