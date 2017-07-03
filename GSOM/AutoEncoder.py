import numpy as np
import time
import sys

class AutoEncoder(object):

    def __init__(self, vis, hid, s = None, m= None, gaussian = False):
        self.w1 =2* (np.random.random((vis, hid)) - 0.5) * 0.00001 #* 0.001
        if gaussian:
            self.w1 = np.random.randn(vis, hid)*s
        if s != None:
            self.w1 = s * np.random.randn(vis, hid)
        # if m != None:
        #     self.w1 += m
        # (np.random.random((784,100)) * 2-1)*0.1
        self.w2 = self.w1.T
        self.b1 = m* np.zeros(hid)#np.random.randn(hid) #* 0.001
        self.b2 = m* np.zeros(vis)#np.random.randn(vis) #* 0.001
        self.st = time.time()
        self.mw1 = np.zeros(self.w1.shape)
        self.mw2 = np.zeros(self.w2.shape)
        self.ma = np.zeros(self.b1.shape)
        self.mb = np.zeros(self.b2.shape)

    def sig(self, X):
        return 1/(1+np.exp(-X))

    def set_params(self, w1, w2, b1, b2):
        # if np.isnan(w1).any():
        #     print 'fuck'
        if w1!= None:
            self.w1 = w1
            self.w2 = w2
            self.b1 = b1
            self.b2 = b2



    def train_batch(self,X, iters, eps, momentum = 0.75, batch_size=10):
        ret = None

        if X.shape[0] > batch_size:
            for i in range(X.shape[0]/batch_size):
                ret =self.train(X[i*batch_size:(i+1)*batch_size], iters, eps * np.exp(-i*1.0/(X.shape[0]/batch_size)), momentum)
                sys.stdout.write('\rbatch %i of %i'%(i+1, X.shape[0]/batch_size))
                sys.stdout.flush()
                momentum *= 0.8
        print ""
        return ret
    def train(self, X, iters, eps, momentum=0.5, wd_param = 0.1):
        d1 = 0.0
        d2 = 0.0
        d3 = 0.0

        for i in range(iters):

            dl_0 = np.random.binomial(1, 1-d1, size=X.shape[0])
            dl_1 = np.random.binomial(1, 1-d2, size = self.b1.shape[0])
            dl_2 = np.random.binomial(1, 1-d3, size = X.shape[0])
            l1 = dl_1*self.sig(np.dot(X*np.array([dl_0]).T, self.w1) + self.b1)  # 1/(1+np.exp(-(np.dot(X,syn0))))
            l2 = np.array([dl_2]).T*self.sig(np.dot(l1, self.w2) + self.b2)  # 1/(1+np.exp(-(np.dot(l1,syn1))))
            l2_delta = (X - l2) * (l2 * (1 - l2))
            l1_delta = l2_delta.dot(self.w2.T) * (l1 * (1 - l1))

            #sparsity :
            avg = np.average(l1, axis=0)
            targ = np.ones(avg.shape) * 0.01
            sparse_term = (- targ/(avg+0.0000000001) + (1-targ)/(1.00000000000001-avg)) * (l1 * (1-l1))
            l1_delta += 0*sparse_term

            dw2 = eps * (l1.T.dot(l2_delta) + momentum *self.mw2 - wd_param*self.w2)
            dw1= eps * (X.T.dot(l1_delta) + momentum*self.mw1 - wd_param*self.w1)
            da= eps * (l1_delta.sum(axis=0) + momentum*self.ma)
            db= eps * (l2_delta.sum(axis=0) + momentum*self.mb )
            self.w2 +=dw2
            self.w1 +=dw1
            self.b1 +=da
            self.b2 +=db
            self.mw1 = dw1
            self.mw2 = dw2
            self.ma = da
            self.mb = db
        return l2

    def predict(self, X):
        l1 = np.matrix(self.sig(np.dot(X, self.w1) + self.b1))  # 1/(1+np.exp(-(np.dot(X,syn0))))
        l2 = self.sig(np.dot(l1, self.w2) + self.b2)
        return l2

