import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import samples_generator
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from bgsom import  GSOM

X1, c1 = samples_generator.make_blobs(n_samples=1000, random_state=1,shuffle=False)
X2, c2 = samples_generator.make_blobs(n_samples=1000, random_state=2, shuffle=False)

X = X1 * X2
# Y = TSNE().fit_transform(X)
Y= GSOM().fit_transform(normalize(X), lr = 1.0, beta=0.75, sf=0.9, wd=0.0005, fd=0.5)#X,lr = 1.0, beta=0.0,sf=0.01, fd=0.75, wd=0.5)

plt.scatter(Y.T[0], Y.T[1], c = c1 )
plt.show()

# a = 10
# b = 10
# y=[]
# x=[]
# z = []
# it = 40
# for i in range(it):
#
#     y.append(a *np.exp(-17.50*i**2/20**2))
#     # b *= np.exp(-0.1*i**2/it**2)
#     # z.append(b)
#     x.append(i)
#
# plt.subplot(211)
# plt.plot(x, y, c = 'blue')
# # plt.subplot(212)
# # plt.plot(x, z, c = 'red')
# plt.show()