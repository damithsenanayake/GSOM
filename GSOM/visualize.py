import matplotlib.pyplot as plt
import numpy as np


def scatter_basic(x, y ):
    plt.scatter(x, y, edgecolors = 'none', alpha=0.7, c='#2e8c68')
    plt.show()
    
def scatter_gradient(x, y):
    plt.scatter(x, y, edgecolors = 'none', alpha=0.7, c = plt.cm.Set1(np.array(range(x.shape[0])).astype(float)/x.shape[0]))
    plt.show()