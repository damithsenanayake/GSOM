import numpy as np

class SOM(object):
    def __init__(self, dims, grid_side):
        self.neurons = np.random.random((grid_side * grid_side, dims))
        grid = []
        # for x in range(grid_side):
        #     for y in range(grid_side):
        #         grid.append([x, y])
        # self.grid = np.array(grid).astype(float)

        self.grid = np.random.random((grid_side * grid_side, 2))


    def find_bmu(self, x):

        diffs = (self.neurons - x)
        dists = np.linalg.norm(diffs, axis=1)

        bmu = np.argmin(dists)

        return bmu


    def get_neighborhood(self, node, num):

        coords = self.grid[node]

        diffs = self.grid - coords
        dists = np.linalg.norm(diffs, axis=1)

        return np.argsort(dists)[:int(num)], np.sort(dists)[:int(num)]

    def train(self, X):
        rad = 50

        for b in range(0,10000):
            for x in X:
                bmu = self.find_bmu(x)
                neighbors, dists = self.get_neighborhood(bmu, rad)
                if dists.shape[0] == 0:
                    return
                m = max(dists)
                h = np.exp(-dists ** 2 / m ** 2)

                self.neurons[neighbors] += (x - self.neurons[neighbors] ) * np.array([h]).T

                neighbors, dists = self.get_neighborhood(bmu, 70)
                m = max(dists)
                h = np.exp(-dists ** 2 / m ** 2)

                self.grid[neighbors] += 0.013*(self.grid[bmu] - self.grid[neighbors])*np.array([h]).T/h.sum()
            rad *= 0.99



