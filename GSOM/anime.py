from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

# ax = p3.Axes3D(fig)
fig = plt.figure(figsize=(12,9))
# create the parametric curve
t=np.arange(0, 2*np.pi, 2*np.pi/100)
x=np.random.random((100, 5))#np.array([np.cos(t), np.sin(t)])
y=np.random.random((x.shape[0], 5))#np.array([np.sin(t), np.cos(t)])

print x[0]
print y[0]
# create the first plot
c = ['red', 'yellow']

odd = np.array([1, 3])
even = np.array([0, 2, 4])

points = plt.plot(x[0][odd], y[0][odd], 'o',c='red' , alpha  =0.5 )[0]
points2 = plt.plot(x[0][even], y[0][even], 'o', c='yellow', alpha=0.5)[0]
# plt.show()
# line, = plt.plot(x, y,  label='parametric curve')
# ax.legend()
# ax.set_xlim([-1.5, 1.5])
# ax.set_ylim([-1.5, 1.5])
# ax.set_zlim([-1.5, 1.5])

# second option - move the point position at every frame
def update_point(n, x, y, points, points2):
    points.set_data(np.array([x[n][odd], y[n][odd]]))
    points2.set_data(np.array([x[n][even], y[n][even]]))
    # point.set_3d_properties(z[n], 'z')
    return points, points2


ani=animation.FuncAnimation(fig, update_point, 5, fargs=(x, y, points, points2))
plt.show()
