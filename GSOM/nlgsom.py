from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

class SOM(object):

    def __init__(self):
        self.w = None
        self.b = None
        self.p = None
        self.Y = None

    def fit(self, X):
        self.Y = np.random.random((X.shape[0], 2))
        self.w = np.random.randn(X.shape[0], X.shape[1])
        self.p = np.zeros((X.shape[0], X.shape[1]))
        self.b = np.zeros(X.shape[0])
        radius = 0.5
        for i in range(100):
            for x in X:
                z = self.w.dot(x) + self.b
                h = sigmoid(z)
                bmu = np.argmin(np.abs(h-0.5))
                Ldist = np.linalg.norm((self.Y[bmu] - self.Y),axis=1)
                neighbors = np.where(Ldist < radius)[0]
                # self.p[bmu] = h
                self.w[neighbors]-= 1* x * np.array([np.exp(-Ldist[neighbors]**2/(2*radius**2))*(h[neighbors])**2*(1-h[neighbors])*(0.5-h[neighbors]) ]).T
                self.b[neighbors]-=1* np.exp(-Ldist[neighbors]**2/(2*radius**2))*(h[neighbors])**2*(1-h[neighbors])*(0.5-h[neighbors])
            radius = 0.5*np.exp(-i / 100)

    def predict(self, X):

        out =[]
        for x in X:
            z = self.w.dot(x) + self.b
            h = sigmoid(z)
            bmu = np.argmin(np.abs(h - 0.5))
            out.append(self.Y[bmu])

        return np.array(out)

    def fit_transform(self, X):
        self.fit(X)
        return self.predict(X)




som = SOM()
X, color = datasets.make_blobs(n_features=5, n_samples=1000, centers=3)

Y = som.fit_transform(X)

plt.scatter(Y.T[0], Y.T[1], s = 15, c = plt.cm.jet(color/(3*1.0)), edgecolors='none', alpha=0.375)

plt.show()