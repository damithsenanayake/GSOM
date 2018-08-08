import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

def sigmoid(x):
    return (1./(1+np.exp(-x)))

x = np.array(range(100)).astype(float)
# y =np.exp(-20.5*((x)/float(np.amax(x)))**(6))#(1.+(x/100.)**2)**-6#sigmoid(x**2/100.**2)#
# z = 0.9**x
x/= 20
print x
# y = np.sqrt(1-x**2)#-(x)**2+1
a = 1
# y = (1 + (x))**-a
y = (1+x**4)**-a# 1-np.exp(-4.*((x)**3))
# z = 1-y
z = np.exp(-0.5*(x)**2)
# z = np.exp(-10.5*(x/float(np.amax(x)))**2)
# plt.plot(x, 1-x)
#
# plt.plot(x, (1-x+(x**6/8)))

plt.plot(x,y)
plt.plot(x, z)
plt.show()
