import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

def sigmoid(x):
    return (1./(1+np.exp(-x)))

x = np.array(range(100)).astype(float)
# y =np.exp(-20.5*((x)/float(np.amax(x)))**(6))#(1.+(x/100.)**2)**-6#sigmoid(x**2/100.**2)#
# z = 0.9**x
x/= x.max()
print x
# y = np.sqrt(1-x**2)#-(x)**2+1
a = 1
# y = (1 + (x))**-a
y = np.exp(-5.*x**6)#np.exp(-x)#(1+x)**-1#(1+(0.1*x**2))**-a# 1-np.exp(-4.*((x)**3))

y*=(1-x)**0.5
# z = 1-y
z = (1+x)**-1#np.exp(-1*x**3)#
# z = np.exp(-10.5*(x/float(np.amax(x)))**2)
# plt.plot(x, 1-x)
#
# plt.plot(x, (1-x+(x**6/8)))
L = 100*np.ones(x.shape)
for i in range(1, L.shape[0]):
    L[i] = L[i-1]*(1-x[i])**0.2
v = z-y
v/=v.max()
w = 1-np.exp(-4.5*x**2)
w/=w.max()
# plt.plot(x,y)
# plt.plot(x, (z))
# # # plt.plot(x, (z-y)/(z-y).max())
# plt.plot(x, z)
plt.plot(x, y)
# plt.plot(x, v)
plt.show()
