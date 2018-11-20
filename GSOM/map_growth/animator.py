import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0,0,1,1], frameon=False)

data = np.random.random((4,2))
scat = ax.scatter(data.T[0], data.T[1])
for i in range(3000):
    fig = plt.figure(figsize=(7, 7))

    X = np.loadtxt(str(i)+'.csv', delimiter=',', dtype=float)
    plt.scatter(X.T[0],X.T[1])
    plt.savefig('../mapgrowth_gradual/'+str(i)+'.png')
# def animate(i):
#     X = np.loadtxt(str(i%25)+'.csv', delimiter=',', dtype=float)
#
#     scat.set_offsets(X)
#     return scat

animation = FuncAnimation(fig, animate, interval=200)
plt.show()
