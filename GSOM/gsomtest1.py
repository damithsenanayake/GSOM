import numpy as np
from GSOM import GSOM
import matplotlib.pyplot as plt
data = np.loadtxt('../data.csv', delimiter=',')

gsom = GSOM(sp=0.5)

gsom.batch_train(data, iter=20)

x = []
y = []
labels = []

for k in range(data.shape[0]):
    bmu = gsom.str_strip(gsom.find_bmu(data[k])[0])[0]
    x.append(bmu[0])
    y.append(bmu[1])

    labels.append(k%2)


plt.scatter(x, y)
for label, x_, y_ in zip(labels, x, y):
    plt.annotate(label, xy=(x_, y_), xytext=(-20, 20), textcoords='offset points', ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
plt.show()