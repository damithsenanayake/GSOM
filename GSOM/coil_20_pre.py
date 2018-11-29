from scipy import misc
import glob
import numpy as np
import matplotlib.pyplot as plt
Y =[]
X = []
for image_path in glob.glob("/home/senanayaked/Part2/NewVolume/coil-20-proc/*.png"):
    image = misc.imread(image_path)
    t = int(image_path.split('/')[-1].split('_')[0].replace('obj', ''))
    # if t in [12, 14, 15, 16, 17, 18, 20]:
    #     continue
    X.append(image.flatten())
    Y.append([t])

X = np.array(X)
Y = np.array(Y)

out = np.concatenate((X, Y), axis=1)

np.savetxt("/home/senanayaked/data/coil20_filtered.csv", out, delimiter=",")