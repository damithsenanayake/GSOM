import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

x = np.array(range(100))
y = np.exp(-15.5*(x/100.)**(1+0.5*50))
plt.plot(x, y)
plt.show()
