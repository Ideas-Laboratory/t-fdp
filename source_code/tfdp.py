import pandas as pd
from scipy.sparse import csr_matrix, tril
import numpy as np
import dask.dataframe  # to read large graph data
from utils import pivotMDS, scaleByEdge, pivot_spd
from optimize.ibFFT_CPU import ibFFT_CPU
from optimize.ibFFT_GPU import ibFFT_GPU

from optimize.Exact import Exact
from optimize.BH import BH
from optimize.RVS import RVS
import torch
import numba


def readgraph(filename, returnDijMat=False):
    edges = dask.dataframe.read_csv(filename, header=0,
                                    names=["src", "tgt", "_"], sep=' ', comment="%", dtype={"src": np.int32, "tgt": np.int32, "_": object})
    startid = min(edges.src.min().compute(), edges.tgt.min().compute())
    rows = edges.src.compute().to_numpy() - startid
    cols = edges.tgt.compute().to_numpy() - startid
    idx = (rows != cols)
    rows = rows[idx]
    cols = cols[idx]
    N = max(rows.max(), cols.max()) + 1
    data = np.ones_like(rows)
    graph = csr_matrix((np.append(data, data), (np.append(
        rows, cols), np.append(cols, rows))), shape=(N, N))

    edgesrc = graph.indptr
    edgetgt = graph.indices
    if N > 41234:
        returnDijMat = False
    graph = tril(graph)
    if returnDijMat:
        p = N
        n = N
        pivot_p = np.arange(n)
        p_vis = np.zeros((p, n), dtype=np.int8)
        p_next = np.zeros((p, n), dtype=np.int32)
        p_spd = np.zeros((p, n), dtype=np.float32)
        pivot_spd(edgesrc, edgetgt, pivot_p, p_spd, p_vis, p_next)
        return graph, edgesrc, edgetgt, p_spd
    return graph, edgesrc, edgetgt, None


# def warmup_cpu():
#     np.random.seed(0)
#     torch.manual_seed(0)
#     combine = True
#     gamma = 2.0
#     beta = 8.0
#     alpha = 0.1
#     max_iter = 300
#     n_interpolation_points = 3  # parameter of FFT，Bigger, more accurate. 3 is enough
#     intervals_per_integer = 1.0  # parameter of FFT，Smaller, more accurate. 1.0 is enough
#     min_num_intervals = 100  # parameter of FFT
#     edgesrc = np.array([0, 1, 1])
#     edgetgt = np.array([1])
#     graph = csr_matrix(
#         (np.array([1]), (np.array([0]), np.array([1]))), shape=(2, 2))
#     pospds = pivotMDS(graph, edgesrc, edgetgt, NP=2, hidden_size=2)
#     pospds = scaleByEdge(pospds, edgesrc, edgetgt) * pospds
#     posres, t = ibFFT_CPU(1.0 * np.array([[0, 1], [2, 3]]), edgesrc, edgetgt, n_interpolation_points, intervals_per_integer, min_num_intervals,
#                           alpha, beta, gamma, max_iter, combine=True, seed=None)


grid_dim = (256,)
block_dim = (128,)


class tFDP:
    def __init__(self, init='pmds', algo='ibFFT_CPU', max_iter=300, alpha=0.1, beta=8.0, gamma=2.0,
                 n_interpolation_points=3, intervals_per_integer=1.0, min_num_intervals=100, combine=True, randseed=None):
        numba.set_num_threads(8)
        self.init = init
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.n_interpolation_points = n_interpolation_points
        self.intervals_per_integer = intervals_per_integer
        self.min_num_intervals = min_num_intervals
        self.combine = combine
        self.randseed = randseed
        self.edgesrc = []
        self.edgetgt = []
        self.algo = algo
        # if (self.algo == 'ibFFT_CPU'):
        #     warmup_cpu()

    def readgraph(self, filename):
        graph, edgesrc, edgetgt, _ = readgraph(filename)
        self.edgesrc = edgesrc
        self.edgetgt = edgetgt
        self.graph = graph
        return graph, edgesrc, edgetgt

    def graphinfo(self):
        print("Graph info: N = ",
              self.graph.shape[0], ", E = ", self.edgetgt.shape[0])

    def optimization(self, graph=None, edgesrc=[], edgetgt=[], init=None, algo=None):
        if (len(edgesrc) == 0 or len(edgetgt) == 0):
            if (len(self.edgesrc) == 0 or len(self.edgetgt) == 0):
                raise "Please readGraph or enter a correct data first"
        else:
            self.edgesrc = edgesrc
            self.edgetgt = edgetgt
        if (graph is None):
            if (self.graph is None):
                raise "Please readGraph or enter a correct data first"
        else:
            self.graph = graph
        if init is not None:
            self.init = init
        if algo is not None:
            self.algo = algo
        graph = self.graph
        edgesrc = self.edgesrc
        edgetgt = self.edgetgt
        if self.init == 'pmds':
            if (self.randseed):
                np.random.seed(self.randseed)
            noise_pos = 0.01 * np.random.randn(graph.shape[0], 2)
            pospds = pivotMDS(graph, edgesrc, edgetgt, NP=100, hidden_size=2)
            pospds *= 2 * scaleByEdge(pospds, edgesrc, edgetgt)
            init = 1.0 * pospds.copy() + noise_pos
        elif type(self.init) is np.ndarray and self.init.shape == (graph.shape[0], 2):
            init = self.init
        n_interpolation_points = 3
        if (self.combine):
            n_interpolation_points = 1
        posres = init
        t = 0
        if (self.algo == 'ibFFT_CPU'):
            posres, t = ibFFT_CPU(init, edgesrc, edgetgt, n_interpolation_points, self.intervals_per_integer, self.min_num_intervals,
                                  self.alpha, self.beta, self.gamma, self.max_iter, self.combine, self.randseed)
        if (self.algo == 'ibFFT_GPU'):
            posres, t = ibFFT_GPU(init, edgesrc, edgetgt, n_interpolation_points, self.intervals_per_integer, self.min_num_intervals,
                                  self.alpha, self.beta, self.gamma, self.max_iter, self.combine, self.randseed)

        if (self.algo == 'Exact'):
            posres, t = Exact(init, edgesrc, edgetgt, self.alpha,
                              self.beta, self.gamma, self.max_iter, self.randseed)
        if (self.algo == 'BH'):
            posres, t = BH(init, edgesrc, edgetgt, self.alpha,
                           self.beta, self.gamma, self.max_iter, self.randseed)
        if (self.algo == 'RVS'):
            posres, t = RVS(init, edgesrc, edgetgt, self.alpha,
                            self.beta, self.gamma, self.max_iter, self.randseed)
        return posres, t
