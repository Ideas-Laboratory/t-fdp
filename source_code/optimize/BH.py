
import torch
import numpy as np
from .Exact_BH_Kernel import *
import bh_tforce
import sys
import time


def BH(pos, edgesrc, edgetgt, alpha=0.1, beta=8, gamma=2, max_iter=300,  seed=None):
    angle = 0.6
    verbose = 9
    n_components = 2
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
    if seed is not None:
        torch.manual_seed(seed)
    for it in range(max_iter):
        AttrForce(attr_force, dC, pos, edgesrc, edgetgt, bias, N, np.float32(
            beta), d3alpha)
        dC += attr_force
        dC -= 0.01 * d3alpha * \
            torch.normal(0, 1, size=pos.shape).numpy().astype(np.float32)
        bh_tforce.tforce_neg_gradient(
            pos, repl_force, angle, d3alpha * paraFactor, n_components, verbose, num_threads=1)
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
