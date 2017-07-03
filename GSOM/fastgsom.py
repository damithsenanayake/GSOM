import numpy as np
from scipy.spatial.distance import minkowski
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale
import sys

class GSOM(object):

    def __init__(self):
       self.nhood = 3
       self.LRdec = 0.1
       self.SF = 0.3
       self.alpha = 0.7
       self.gamma = 0.5
       self.LR = 1
       self.Herr = 0
       self.QE = 0
       self.p = np.array([[0, 0], [0, 1], [1, 0], [0, 1]]) # node positions
       self.n = None
       self.D = None
       
    def fit(self, X, iterations=100, alpha=0.7, gamma=0.5, SF=0.3, LR=1, LRdec=0.1, nhood=3):
        
        self.n = X.shape[0]
        self.D = X.shape[1]
        n = self.n
        D = self.D

        self.W = np.random.random(4, D)
        self.GT = -D * np.log(SF)
        
        self.c = np.zeros((self.p.shape(0), 5))
        
        for i in range(self.p.shape[0]):
            self.c[i] = np.array([i, self.find_neighbors(i)])


        while LR >= 0.1:
            Herr = 0
            e = np.zeros(self.p.shape[0])
            for i in range(n):
                minInd = np.argmin(np.linalg.norm(self.W - X[i], axis=1))
                N = self.c[minInd]
                e[minInd] += np.linalg.norm(self.W[minInd] - X[i])
                if e[minInd] > Herr:
                    Herr = e[minInd]
                self.QE = np.sum(e)

                notnaninds = N[not np.isnan(N)]

                self.W[notnaninds] += (X[i] - self.W[notnaninds]) * LR

                if Herr > self.GT:
                    if np.sum(np.isnan(N)): # winner is a boundary node
                        naninds = np.where(np.isnan(N))[0]

                        if 1 in naninds:
                            self.p.append(np.array(self.p[minInd] + np.array([0, 1])))
                            for k in range(self.p.shape[0]):
                                self.c[k, 2:] = self.find_neighbors(k)
                            self.W.append(self.getNewWeight())

                        if 2 in naninds:
                            self.p.append(np.array(self.p[minInd] + np.array([0, -1])))

                            for k in range(self.p.shape[0]):
                                self.c[k, 2:] = self.find_neighbors(k)
                            self.W.append(self.getNewWeight())

                        if 3 in naninds:
                            self.p.append(np.array(self.p[minInd] + np.array([-1, 0])))

                            for k in range(self.p.shape[0]):
                                self.c[k, 2:] = self.find_neighbors(k)

                            self.W.append(self.getNewWeight())

                        if 4 in naninds:
                            self.p.append(np.array(self.p[minInd] + np.array([1, 0])))

                            for k in range(self.p.shape[0]):
                                self.c[k, 2:] = self.find_neighbors(k)
                            self.W.append(self.getNewWeight())


###########################################################
    def find_neighbors(self, i):


        top = np.where(self.c[:, 0]==self.p[i][0] and self.c[:, 1] == self.p[i][1]+1)[0]
        bottom = np.where(self.c[:, 0] == self.p[i][0] and self.c[:, 1] == self.p[i][1]-1)[0]
        left = np.where(self.c[:, 0] == self.p[i][0]-1 and self.c[:, 1] == self.p[i][1])[0]
        right = np.where(self.c[:, 0] == self.p[i][0]+1 and self.c[:, 1] == self.p[i][1])[0]

        if not np.shape(top)[0]:
            top = np.nan
        if not np.shape(bottom)[0]:
            bottom = np.nan
        if not np.shape(left)[0]:
            left = np.nan
        if not np.shape(right)[0]:
            right = np.nan

        return np.array([top, bottom, left, right])



    
