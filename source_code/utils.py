
from numba import njit, prange
import numpy as np
from numba import prange, njit


@njit(parallel=False, cache=True)
def getC(C, d, n, k, deltaIS, deltaRJ, sumALL):
    for i in prange(n):
        for j in range(k):
            C[i][j] = -1.0/2 * (d[j][i]**2 - deltaRJ[j] - deltaIS[i] + sumALL)
    return C


def power_iteration(A, num_simulations: int):
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm

    return b_k


@njit(parallel=False, cache=True)
def scaleByEdge(pos, edgesrc, edgetgt):
    N = edgesrc.shape[0] - 1
    stop_np = 0.0
    sbot_np = 0.0
    for i in prange(N):
        for k in range(edgesrc[i], edgesrc[i+1]):
            j = edgetgt[k]
            mag = np.sqrt((pos[i][0] - pos[j][0])**2 +
                          (pos[i][1] - pos[j][1])**2)
            stop_np += mag
            sbot_np += mag * mag
    return stop_np/sbot_np


def pivotMDS(graph, edgesrc, edgetgt, NP=200, hidden_size=128):
    p = NP
    n = graph.shape[0]
    if p >= n:
        p = n
        pivot_p = np.arange(n)
    else:
        pivot_p = np.random.choice(np.arange(n), p, replace=False)
    p_vis = np.zeros((p, n), dtype=np.int8)
    p_next = np.zeros((p, n), dtype=np.int32)
    d = np.zeros((p, n), dtype=np.float32)
    pivot_spd(edgesrc, edgetgt, pivot_p, d, p_vis, p_next)
    d[np.isinf(d)] = 0
    (k, n) = d.shape
    d2 = d**2
    deltaIS = d2.sum(axis=0)/k
    deltaRJ = d2.sum(axis=1)/n
    sumALL = d2.sum()/(n*k)
    # NP = n_p
    C = np.zeros((n, NP), dtype=np.float64)
    C = getC(C, d, n, k, deltaIS, deltaRJ, sumALL)
    B = np.dot(C.T, C)
    pos = np.zeros((n, hidden_size))
    for i in range(hidden_size):
        V = power_iteration(B, 100).reshape(1, -1)
        lbd = np.dot(V, np.dot(B, V.T))
        pos[:, i] = np.dot(C, V.reshape(-1, 1)).reshape(-1)
        B = B - lbd / np.linalg.norm(V)**2 * np.dot(V.T, V)
    return pos


@njit(parallel=True, cache=True)  # For calculate Stress Error
def pivot_spd(indptr, indice, pivot, p_spd, p_vis, p_next):
    num_p = pivot.shape[0]
    for i in prange(num_p):
        l = 0
        r = 1
        p_next[i][0] = pivot[i]
        p_vis[i][pivot[i]] = 1
        p_spd[i][pivot[i]] = 0.0
        while (l < r):
            p = p_next[i][l]
            for np in indice[indptr[p]:indptr[p+1]]:
                if not p_vis[i][np]:
                    p_vis[i][np] = 1
                    p_next[i][r] = np
                    r += 1
                    p_spd[i][np] = p_spd[i][p] + 1

            l += 1
        l += 0
        r += 0
