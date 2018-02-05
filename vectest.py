import matplotlib.pyplot as plt
from sklearn import datasets

X, t = datasets.make_s_curve(d, random_state=1)#np.random.random((100, 3))

from GSOM.manifoldgsom import SOM

Y = SOM().fit_transform(X)

plt.scatter(Y.T[0], Y.T[1], c= t, cmap=plt.cm.Spectral)
plt.show()