import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tsne = np.array(pd.read_csv('tem-2000-mnist.csv'))
tem = np.array(pd.read_csv('tsne-2000-mnist.csv'))
xticks = np.array(['ARS', 'AMI', 'ARS', 'AMI'])


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Cluster Quality Metric')
ax.set_ylabel('Score')
# bp1 = ax.boxplot(tem, labels=np.array(['', 'ARS', '', 'AMI']),showfliers=False)
# ax.boxplot(tsne, labels=np.array(['ARS', '', 'AMI', '']),  vert=1, whis=1.5, notch=0, sym='+',showfliers=False)
vals = np.concatenate((tsne, tem), axis=1)
x =np.array([0,0.5,  2, 2.5])
zeros = np.zeros(x.shape)+0.4
y = vals[-1]
ax.scatter(x, y, c = ['blue', 'blue', 'red', 'red'], marker='o')
ax.errorbar(x, y, c = 'black', yerr=[y-np.min(vals, axis=0), np.max(vals, axis=0)-y] , fmt='+', alpha = 0.5)
# ax.scatter(x, zeros, s = 0)
# plt.setp(bp1['boxes'], color = 'blue')
# plt.setp(bp1['whiskers'], color = 'blue')
# plt.setp(bp1['caps'], color = 'blue')
# plt.setp(bp1['medians'], color = 'green')
# plt.setp(bp1['fliers'], color = 'red', marker = '+')
plt.text(0.25, 0.5, 'DS-GSOM')
plt.text(2.25, 0.5, 't-SNE')
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.xticks(x, xticks)

plt.show()