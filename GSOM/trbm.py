import theano.tensor as T
import theano
import numpy as np

class RBM(object):

    def __init__(self):
        nhid = 192
        nvis = 784
        self.hid = nhid
        self.vis = nvis
        w = theano.shared(np.random.randn(nvis, nhid), 'w')
        self.w = w
        vb = np.random.randn(784)
        va = np.ranom.randn(192)
        visbias = theano.shared(vb, 'vb')
        hidbias = theano.shared(va, 'vh')
        self.vb = visbias
        self.hb = hidbias


    def positive(self, x):
        self.ph = T.nnet.sigmoid(T.dot(x,self.w) + self.vb)
        self.posprods = T.dot(x.T, self.ph)
        self.poshidacts = self.ph.sum(axis = 0)
        self.posvisacts = x.sum(axis = 0)
        self.poshidstates = np.random.binomial(1, self.ph)
        if isinstance(self.poshidstates, int):
            self.poshidstates = np.array([self.poshidstates])

