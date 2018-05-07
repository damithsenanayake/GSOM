import os
import matplotlib.pyplot as plt
import numpy as np

nsamples = 6000

os.system('scp -i ~/nectar/firstpair.pem ubuntu@43.240.98.97:~/GSOM/GSOM/mnist_%i.csv ./scp_transfers/mnist_%i.csv'%(nsamples, nsamples))

d = np.loadtxt('./scp_transfers/mnist_%i.csv'%nsamples)

x, y , labels = d.T

plt.scatter(x, y, edgecolors='none',c=plt.cm.jet(labels/10.0), alpha = 0.5, s = 15)

plt.show()
