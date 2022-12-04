
import torch
import numpy as np
from .Exact_BH_Kernel import computebias, AttrForce, ApplyForce
import sys
import time
from numba import njit

###
# Modified from gove's code.
# d3-force-sampled.js
###


@njit(cache=True)  # L37   function addRandomNode(node)
def addRandomNode(nodeIdx, pos, nearestNodeIdxs, nearestNodeLen, numNeighbors):
    n = pos.shape[0]
    randIdx = np.random.randint(n)
    randDist = np.sqrt(((pos[randIdx] - pos[nodeIdx])**2).sum())
    maxDist = -np.inf
    maxI = 0
    if randIdx in nearestNodeIdxs[nodeIdx]:
        return
    if nearestNodeLen[nodeIdx] < numNeighbors:
        nearestNodeIdxs[nodeIdx][nearestNodeLen[nodeIdx]] = randIdx
        nearestNodeLen[nodeIdx] += 1
        return
    for i in range(nearestNodeLen[nodeIdx]):
        currIdx = nearestNodeIdxs[nodeIdx][i]
        currDist = np.sqrt(((pos[currIdx] - pos[nodeIdx])**2).sum())
        if (currDist > maxDist):
            maxI = i
            maxDist = currDist
    if (randDist < maxDist):
        nearestNodeIdxs[nodeIdx][maxI] = randIdx


@njit(cache=True)  # L73 function getRandIndices(indices, num)
def getRandIndices(indices, num, n):
    cnt = n - num
    for i in range(n-1, cnt-1, -1):
        randIdx = np.random.randint(i+1)
        temp = indices[randIdx]
        indices[randIdx] = indices[i]
        indices[i] = temp
    return indices[cnt:]


@njit(cache=True)  # L92 function approxRepulse(node)
def approxRepulse(pos, repl_force, nodeIdx, randIndices, d3alpha, a, c):
    nrand = randIndices.shape[0]
    for i in range(nrand - 1, -1, -1):
        currNodeIdx = randIndices[i]
        if currNodeIdx == nodeIdx:
            continue
        mvx = pos[currNodeIdx][0] - pos[nodeIdx][0]
        mvy = pos[currNodeIdx][1] - pos[nodeIdx][1]
        dsqare = mvx * mvx + mvy * mvy
        R = d3alpha * a * np.power(1.0 + dsqare, -c)
        repl_force[nodeIdx][0] -= R * mvx
        repl_force[nodeIdx][1] -= R * mvy


@njit(cache=True)  # L126 function constantRepulse(node)
def constantRepulse(pos, repl_force, nodeIdx, numNeighbors, nearestNodeIdxs, nearestNodeLen, d3alpha, a, c):
    addRandomNode(nodeIdx, pos, nearestNodeIdxs, nearestNodeLen, numNeighbors)
    l = nearestNodeLen[nodeIdx]
    for i in range(l-1, -1, -1):
        currNodeIdx = nearestNodeIdxs[nodeIdx][i]
        if currNodeIdx == nodeIdx:
            continue
        mvx = pos[currNodeIdx][0] - pos[nodeIdx][0]
        mvy = pos[currNodeIdx][1] - pos[nodeIdx][1]
        dsqare = mvx * mvx + mvy * mvy
        R = d3alpha * a * np.power(1.0 + dsqare, -c)
        repl_force[nodeIdx][0] -= R * mvx
        repl_force[nodeIdx][1] -= R * mvy


@njit(cache=True)  # L162 function force(_)
def repl(pos, repl_force, nearestNodeIdxs, nearestNodeLen, indicesRepl, numNeighbors, prevIndex, numUpdate, numSamples, d3alpha, a, c):
    i = 0
    j = prevIndex
    n = pos.shape[0]
    upperIndex = prevIndex + numUpdate
    while (i < n) or (j < upperIndex):
        if (j < upperIndex):
            randIndices = getRandIndices(indicesRepl, numSamples, n)
            approxRepulse(pos, repl_force, j % n, randIndices, d3alpha, a, c)
        if (i < n):
            constantRepulse(pos, repl_force, i, numNeighbors,
                            nearestNodeIdxs, nearestNodeLen, d3alpha, a, c)
        i += 1
        j += 1
    prevIndex = int(upperIndex % n)
    return prevIndex


@njit(cache=True)  # function initialize() L189-194
def nearestNodeInit(n, pos, nearestNodeIdxs, nearestNodeLen, numNeighbors):
    for i in range(n):
        while nearestNodeLen[i] < numNeighbors:
            addRandomNode(i, pos, nearestNodeIdxs,
                          nearestNodeLen, numNeighbors)


def RVS(pos, edgesrc, edgetgt, alpha=0.1, beta=8, gamma=2, max_iter=300,  seed=None):
    paraFactor = 1.0
    if (alpha != 0):
        paraFactor /= alpha
        # for keeping a same long range force
    d3alpha = 1.0
    d3alphaMin = 0.01
    E = edgetgt.shape[0]
    N = len(pos)
    if E/N >= 15.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0  # a small d3alpha for higher average degrees
    if E/N >= 50.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0
    pos = np.array(pos, dtype=np.float32)
    st = time.time()
    dC = np.zeros((N, 2), dtype=np.float32)
    attr_force = np.zeros((N, 2), dtype=np.float32)
    repl_force = np.zeros((N, 2), dtype=np.float32)
    pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min())/2
    pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min())/2
    edgesrc = np.array(edgesrc, dtype=np.int32)
    edgetgt = np.array(edgetgt, dtype=np.int32)
    bias = np.zeros(edgetgt.shape[0], dtype=np.float32)
    computebias(N, edgesrc, edgetgt, bias)

    # L171 initialize() for RVS initialization
    numNeighbors = 20
    numNeighbors = min(numNeighbors, N)
    numUpdate = np.ceil(np.power(N, 0.75))
    numSamples = np.ceil(np.power(N, 0.25))

    alpha = 1.0
    indicesRepl = np.arange(N)
    nearestNodeIdxs = np.zeros((N, numNeighbors), dtype=np.int32)
    nearestNodeLen = np.zeros(N, dtype=np.int32)

    nearestNodeInit(N, pos, nearestNodeIdxs, nearestNodeLen, numNeighbors)
    prevIndex = 0

    if seed is not None:
        torch.manual_seed(seed)
    for it in range(max_iter):
        AttrForce(attr_force, dC, pos, edgesrc, edgetgt, bias, N, np.float32(
            beta), d3alpha)
        dC += attr_force
        dC -= 0.01 * d3alpha * \
            torch.normal(0, 1, size=pos.shape).numpy().astype(np.float32)
        repl_force *= 0
        prevIndex = repl(pos, repl_force, nearestNodeIdxs, nearestNodeLen, indicesRepl, numNeighbors,
                         prevIndex, numUpdate, numSamples, d3alpha, paraFactor, gamma)
        dC += repl_force
        ApplyForce(dC, pos, N)
        # 1 - pow(0.02, 1 / 300) = 0.012955423246736264
        d3alpha += (d3alphaMin-d3alpha)*0.012955423246736264
        pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min())/2
        pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min())/2
        # move to center
        dC *= 0.6
        if ((it+1) % 5 == 0):
            print(".", end="")
            sys.stdout.flush()
    print("\n", end="")
    ed = time.time()
    return pos, ed-st
