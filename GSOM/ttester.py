import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

def sigmoid(x):
    return (1./(1+np.exp(-x)))

x = np.array(range(100)).astype(float)#np.random.randn(100).astype(float)

## y =np.exp(-20.5*((x)/float(np.amax(x)))**(6))#(1.+(x/100.)**2)**-6#sigmoid(x**2/100.**2)#
# z = 0.9**x
# dix = 10
#
# xs = x/x.max()
# xd = xs[dix]
# xs -= xs[dix]
# xs /= xd
x/=x.max()
# x/=x[8]



y =np.exp(-3.8*x**2)
# y = y/y.max()
# z = (1+x)**-1

# z-=z.min()
# z/= z.max()01131ds

# print x
# for i in range(1, 15):
#     y = np.exp(-i*(1-x)**2)#-(x)**2+1
#     y-=y.min()
#     y/=y.max()
#     a = 0.08
#     # z = np.exp(-15.*x**2)
#     # z[z<0.6]=0
#     # # # plt.plot(x, z)
#     # y *= a
#     # y += z
#     plt.plot(x, y)
# plt.show(block=False)
# fig=plt.figure()
# for i in range(1, 8):
#     y = np.exp(-15*(1-x)**i)#-(x)**2+1
#     y-=y.min()
#     y/=y.max()
#     a = 0.08
#     # z = np.exp(-15.*x**2)
#     # z[z<0.6]=0
#     # # # plt.plot(x, z)
#     # y *= a
#     # y += z
plt.plot(x, y)
# plt.plot(x, z)
# plt.plot(x,(z-y)*z)
# plt.plot(x, z)
# plt.plot(x, y*z/(y*z).max())
# plt.plot(x, v)
plt.show()
