import numpy as np
from GSOM import GSOM
import matplotlib.pyplot as plt
import pandas as pd


def visualizeW1(opt_W1, vis_patch_side, hid_patch_side, title):
    """ Add the weights as a matrix of images """
    plt.title(title)
    figure, axes = plt.subplots(nrows=hid_patch_side,
                                ncols=hid_patch_side)
    index = 0

    for axis in axes.flat:
        """ Add row of weights as an image to the plot """

        image = axis.imshow(opt_W1[index, :].reshape(vis_patch_side, vis_patch_side),
                            cmap=plt.cm.gray, interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """

    plt.show()


data = pd.read_csv('../mnist_train.csv', header=None).as_matrix()

train = data[:, 1:]

train[train > 0] = 1
t_x = train[:10000]
### Bitmap  indexing for the number of legs

gsom = GSOM(sp=0.999999)

gsom.batch_train(t_x, iter=1)

x = []
y = []
labels = []
for key in gsom.neurons.keys():
    x.append(int(key.split('x')[0]))
    y.append(int(key.split('x')[1]))
    labels.append(gsom.neurons[key].hits)

plt.scatter(x, y)
for label, x_, y_ in zip(labels, x, y):
    plt.annotate(label, xy=(x_, y_), xytext=(-20, 20), textcoords='offset points', ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
plt.show()
x = []
y = []
labels = []

data = pd.read_csv('../mnist_test.csv', header=None).as_matrix()
test_x = data[:, 1:]
for k in range(150):
    bmu = gsom.str_strip(gsom.find_bmu(test_x[k])[0])[0]
    x.append(bmu[0])
    y.append(bmu[1])

    labels.append(data[k][0])  # data[k][-1])

# for i in range(1000):
#     v = train[i]
#     labels.append(data[i][0])
#     bmu = gsom.find_bmu(v)
#     x.append(bmu[0].split('x')[0])
#     y.append(bmu[0].split('x')[1])

colors = []
# for i in range(10) :
#     w = gsom.w[gsom.w.keys()[i]].rbm.w.T
#     visualizeW1(w, 28, 14, gsom.w.keys()[i])

#
# i = 0
# for v in train:
#     bmu = gsom.find_bmu(v)[0]
#     x.append(bmu.split('x')[0])
#     y.append(bmu.split('x')[1])
#     if i %2 == 0:
#         colors.append('green')
#     else:
#         colors.append('red')
#
#     i +=1

plt.scatter(x, y)
for label, x_, y_ in zip(labels, x, y):
    plt.annotate(label, xy=(x_, y_), xytext=(-20, 20), textcoords='offset points', ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
plt.show()
