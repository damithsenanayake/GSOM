from sgsom import GSOM
from AutoEncoder import AutoEncoder
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from AETEST import visualizeW1

fdat = pd.read_csv('/home/senanayaked/data/mnist_train.csv', header=None)
x = np.array(fdat)[:10000, 1:]
X = np.array(x).astype(float)/255.0

gsom = GSOM(784, 0.8, 0.8, 2000, 100, 5)

gsom.train_batch(X)

grid = gsom.predict(X)
