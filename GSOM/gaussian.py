import numpy as np
import matplotlib.pyplot as plt

x = np.array(range(0, 100, 1)).astype(float)
y = 1/ (1+ 0.01*(50-x)**2)
plt.plot(x, y)
print y
plt.show()