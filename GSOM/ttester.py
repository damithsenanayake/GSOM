import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

def sigmoid(x):
    return (1./(1+np.exp(-x)))

x1 = np.array(range(100)).astype(float)#np.random.randn(100).astype(float)
x1/=x1.max()
x = np.random.random(100)
x -= x.min()
## y =np.exp(-20.5*((x)/float(np.amax(x)))**(6))#(1.+(x/100.)**2)**-6#sigmoid(x**2/100.**2)#
# z = 0.9**x
# dix = 10
#
# xs = x/x.max()
# xd = xs[dix]
# xs -= xs[dix]
# xs /= xd
x/=x.max()
x.sort()
# x/=x[8]



y =np.exp(-7*(1-x)**2)
z = 1- np.exp(-4.5 * x ** 4)
# y = y/y.max()
y1 = np.exp(-7*(1-x1)**2)
z1 = 1 - np.exp(-5 * x1 ** 4)

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
plt.scatter(x, y, c = plt.cm.jet(x))
plt.scatter(x, z, c = plt.cm.inferno(x))

# plt.scatter(x1, y1, c= 'red')
# plt.scatter(x1, z1, c = 'red')
# plt.plot(x,(z-y)*z)
# plt.plot(x, z)
# plt.plot(x, y*z/(y*z).max())
# plt.plot(x, v)
plt.show()
