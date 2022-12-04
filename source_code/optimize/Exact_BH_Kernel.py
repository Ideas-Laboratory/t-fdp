from numba import njit, prange, jit
import numpy as np


@njit(parallel=False, cache=True)
def ReplForce(repl_force, pos, N, a, c, alpha):
    for i in prange(N):
        repl_force[i][0] = 0
        repl_force[i][1] = 0
        for j in range(N):
            mv0 = pos[i][0] - pos[j][0]
            mv1 = pos[i][1] - pos[j][1]
            dsqare = mv0 * mv0 + mv1 * mv1
            R = alpha * a * np.power(1.0 + dsqare, -c)
            repl_force[i][0] += R * mv0
            repl_force[i][1] += R * mv1


@njit(parallel=False, cache=True)
def ApplyForce(dC, pos, N):
    for i in prange(N):
        mv0 = dC[i][0]
        mv1 = dC[i][1]
        d = np.sqrt(mv0 * mv0 + mv1 * mv1) + 1e-32
        R = min(d, 1.0)
        pos[i][0] += mv0 / d * R
        pos[i][1] += mv1 / d * R


@njit(parallel=False, cache=True)
def computebias(N, edgesrc, edgetgt, bias):
    for i in prange(N):
        cntSrc = edgesrc[i + 1] - edgesrc[i]
        for k in range(edgesrc[i], edgesrc[i+1]):
            j = edgetgt[k]
            cntTgt = edgesrc[j + 1] - edgesrc[j]
            bias[k] = cntTgt / (cntSrc + cntTgt)


@njit(parallel=False, cache=True)
def AttrForce(attr_force, dC, pos, edgesrc, edgetgt, bias, N, a1, alpha):
    N = attr_force.shape[0]
    for i in prange(N):
        cntSrc = edgesrc[i + 1] - edgesrc[i]
        tempforce = 0.0
        tempforce1 = 0.0
        mvix = pos[i][0]
        mviy = pos[i][1]
        dcix = dC[i][0]
        dciy = dC[i][1]
        constant = np.float32(1.0)
        constant2 = np.float32(0.5)
        for k in range(edgesrc[i], edgesrc[i+1]):
            j = edgetgt[k]
            mv0 = mvix + dcix
            mv1 = mviy + dciy
            b = bias[k]
            mv0 -= (dC[j][0] + pos[j][0])
            mv1 -= (dC[j][1] + pos[j][1])
#             b = b/c
            R = a1 / (constant + mv0 * mv0 + mv1 * mv1) + b
            tempforce -= R*mv0
            tempforce1 -= R*mv1
        mv0 = alpha * tempforce
        mv1 = alpha * tempforce1
        d = np.sqrt(mv0 * mv0 + mv1 * mv1) + 1e-12
        R = min(d, 2000.0)
        attr_force[i][0] = mv0 / d * R
        attr_force[i][1] = mv1 / d * R
