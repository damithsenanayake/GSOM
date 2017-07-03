import numpy as np
from RBM import RBM
import sys
class Neuron(object):
    def __init__(self, gen = 0, gt=0):
        self.rbm = RBM(784, 196)
        self.gen = gen
        self.r_error = gt/2
        self.lasthit = 0
        self.hits = 0
    def train_single(self, x, eps):
        self.hits += 1
        self.rbm.train(x, eps=eps )
        self.w = self.rbm.w
        self.lasthit += 1
        # self.r_error += np.linalg.norm(self.rbm.reconstruct()[0] - x)
        return np.linalg.norm(self.rbm.reconstruct()[0] - x)#self.r_error
    def reconstruct(self, x):
        #self.rbm.positive(np.array([x]))  self.rbm.disc_error(x)
        return np.linalg.norm(self.rbm.reconstruct() - x)

class GSOM(object):
    def __init__(self, dims = 784, sp = 0.9):
        self.GT = -1 * dims * np.log(sp)
        self.max_age = 0
        self.gen = 0
        self.eps = 0.01
        self.dim = dims
        self.neurons = {}
        self.hit_timer = 0

        self.range = 1.1
        self.fd = 0.75
        for i in range(2):
            for j in range(2):
                self.neurons[str(i) + 'x' + str(j)] = Neuron(gen=0, gt= self.GT)

    def find_bmu(self, x):
        e = float('Inf')
        cand = None
        for n_i in self.neurons.keys():
            e_i = self.neurons[n_i].reconstruct(np.array([x]))
            if e_i < e:
                e = e_i
                cand = n_i
        return cand, e

    def batch_train(self, x, iter = 10):
        print 'batch training starting'
        for i in range(1,iter+1):
            dead = 0
            cut = 0
            prune_coef = 0.9
            if x.shape[0] > 10:
                batches = x.shape[0] /10
                b = -1 * np.log(prune_coef) / (batches + 1)

                for j in range(batches):
                    # cut = 1+prune_coef - np.exp(-1 * b * j)
                    sys.stdout.flush()

                    sys.stdout.write("\r epoch:%i/%i batch:%i/%i n : %i "%(i, iter, j+1, batches, len(self.neurons)))
                    xk = x[j*10:j*10+10]
                    self.gen +=1
                    for xi in xk:
                        self.hit_timer += 1
                        self.train(xi)

                    cutoff_age =  self.gen/2#np.log2(self.gen)#/(i*iter)
                    if len(self.neurons.keys()) > 70:
                        for k in self.neurons.keys():

                            if self.neurons[k].lasthit < self.hit_timer*0.7:
                                del self.neurons[k]
                                dead +=1
                                continue
                            if self.neurons[k].gen < cutoff_age:
                                del self.neurons[k]
                                dead += 1
                                continue
                            if self.neurons[k].hits < 0.02 *(j/100+1) * np.log2(x.shape[0]*i/iter):
                                cut += 1
                                del self.neurons[k]
                                continue
                            # if self.w[k].r_error < self.GT :
                            #     del self.w[k]
                            #     cut +=1
                    self.range *= 0.9

                    self.eps *= (0.9 * (1- 3.8 / len(self.neurons)))
            print '\nepoch :', i, ' nodes : ', len(self.neurons.keys()), 'dead: ', dead, 'cut :', cut
            t = 0

            # for k in self.w.keys():
            #     if self.w[k].r_error > self.GT * 0.9:
            #         del self.w[k]
            #         t+=1
            # print 'deleted nodes: ', t

    def train(self, x):
        bmu, e = self.find_bmu(x)
        r_error = self.neurons[bmu].r_error
        self.neurons[bmu].lasthit = self.hit_timer
        for v in self.inrange(bmu):
            n_i = v[0]
            hn = float(v[1])
            eps = self.eps *hn
            error = self.neurons[n_i].train_single(np.array([x]), eps=eps)
            if n_i == bmu:
                self.neurons[bmu].r_error += error
        if r_error > self.GT and len(self.neurons) < 150:
            self.grow(bmu)

    def str_strip(self, string):
        return np.array([string.split('x')]).astype(int)

    def str_dress(self, arr):
        return str(arr.flatten()[0]) + 'x' + str(arr.flatten()[1])

    def inrange(self, bmu):
        out_array = []
        for ni in self.neurons.keys():
            dist = np.linalg.norm(self.str_strip(bmu) - self.str_strip(ni))
            lr = (dist/self.range)**2
            if (dist< self.range):
                out_array.append([ni, lr])

        return np.array(out_array)

    def grow(self, n_i):

        p = self.str_strip(n_i)

        up = p + np.array([0, -1])
        down = p + np.array([0, +1])
        left = p + np.array([-1, 0])
        right = p + np.array([+1, 0])

        neighbors = np.array([up, right, down, left])
        direction = 0
        for nei in neighbors:
            try:
                self.neurons[self.str_dress(nei)].r_error += self.neurons[n_i].r_error * self.fd
            except KeyError:
                new_b = self.type_b(nei, direction)
                new_a = self.type_a(nei, direction)
                new_c = self.type_c(nei, direction)

                if new_b.all() == 0:
                    if new_a.all() == 0:
                        if new_c.all() == 0:
                            w = np.ndarray(shape=(self.dim))
                            w.fill(0.5)
                        else:
                            w = new_c
                    else:
                        w = new_a
                else:
                    w = new_b

                neu = Neuron(gen=self.gen, gt = self.GT)
                self.max_age = self.gen
                neu.w = w
                self.neurons[self.str_dress(nei)] = neu

            direction += 1

        self.neurons[n_i].res_err = self.GT / 2

    def type_b(self, nei, direction):
        try:
            if direction == 0 or direction == 2:
                return (self.neurons[self.str_dress(nei + np.array([0, -1]))].rbm.w + self.neurons[
                    self.str_dress(nei + np.array([0, 1]))].rbm.w) * 0.5
            return (self.neurons[self.str_dress(nei + np.array([-1, 0]))].rbm.w + self.neurons[
                self.str_dress(nei + np.array([1, 0]))].rbm.w) * 0.5

        except KeyError:
            return np.array([0])

    def type_a(self, nei, direction):
        try:
            anc = {0: np.array([0, -1]),
                   1: np.array([1, 0]),
                   2: np.array([0, 1]),
                   3: np.array([-1, 0])}
            w1 = self.neurons[self.str_dress(nei + anc[direction])].rbm.w
            w2 = self.neurons[self.str_dress(nei + 2 * anc[direction])].rbm.w
            return 2 * w1 - w2
        except KeyError:
            return np.array([0])

    def type_c(self, nei, direction):
        try:
            anc = {0: np.array([0, -1]),
                   1: np.array([1, 0]),
                   2: np.array([0, 1]),
                   3: np.array([-1, 0])}

            if direction == 0 or direction == 2:
                try:
                    return 2 * self.neurons[self.str_dress(nei + anc[direction])].rbm.w - self.neurons[
                        self.str_dress(nei + anc[direction] + np.array([1, 0]))].rbm.w
                except KeyError:
                    return 2 * self.neurons[self.str_dress(nei + anc[direction])].rbm.w - self.neurons[
                        self.str_dress(nei + anc[direction] + np.array([-1, 0]))].rbm.w

            else:
                try:
                    return 2 * self.neurons[self.str_dress(nei + anc[direction])].rbm.w - self.neurons[
                        self.str_dress(nei + anc[direction] + np.array([0, 1]))].rbm.w
                except KeyError:
                    return 2 * self.neurons[self.str_dress(nei + anc[direction])].rbm.w - self.neurons[
                        self.str_dress(nei + anc[direction] + np.array([0,-1]))].rbm.w
        except KeyError:
            return np.array([0])
