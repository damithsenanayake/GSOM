from AutoEncoder import AutoEncoder
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def visualizeW1(opt_W1, vis_patch_side, hid_patch_side):

    """ Add the weights as a matrix of images """

    figure, axes = plt.subplots(nrows = hid_patch_side,
                                              ncols = hid_patch_side)
    index = 0

    for axis in axes.flat:

        """ Add row of weights as an image to the plot """

        image = axis.imshow(opt_W1[index, :].reshape(vis_patch_side, vis_patch_side),
                            cmap = plt.cm.gray, interpolation = 'nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """

    plt.show()


fdat = pd.read_csv('/home/senanayaked/data/mnist_train.csv', header=None)
x = np.array(fdat)[:10000, 1:]
X = np.array(x).astype(float)/255.0

st = time.time()

ae = AutoEncoder(vis=784, hid=100, gaussian=True)

Y=ae.train_batch(X, 400, 0.00075,batch_size=
                 100)
ela = time.time()-st
print ela /1000
plt.imshow(np.reshape(Y[0]*255, (28,28)))
plt.show()

visualizeW1(ae.w1.T, 28, 10)