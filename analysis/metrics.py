from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from numba import jit, prange, njit
import cupy as cnp

from numba import njit, prange


@njit(parallel=True, cache=True)
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


@njit(parallel=True, cache=True)
def bestScale_np(p_spd, pos, pivot_p):
    p, N = p_spd.shape
    stop_np = 0.0
    sbot_np = 0.0
    for k in prange(p):
        i = pivot_p[k]
        for j in range(N):
            if (p_spd[k][j] == 0):
                continue
            mag = np.sqrt((pos[i][0] - pos[j][0])**2 +
                          (pos[i][1] - pos[j][1])**2)
            stop_np += mag/p_spd[k][j]
            sbot_np += (mag/p_spd[k][j])**2
    return stop_np/sbot_np


@njit(parallel=True, cache=True)
def getSE_np(p_spd, pivot_p, pos, res_np):
    p, N = p_spd.shape
    for k in prange(p):
        res_np[k] = 0.0
        ni = 0
        i = pivot_p[k]
        for j in range(N):
            if (p_spd[k][j] == 0):
                continue
            mag = np.sqrt((pos[i][0] - pos[j][0])**2 +
                          (pos[i][1] - pos[j][1])**2)
            res_np[k] += 1/p_spd[k][j]**2 * (mag-p_spd[k][j])**2
            ni += 1


def SE(pos, edgesrc, edgetgt):
    n = edgesrc.shape[0] - 1
    p = int(512*1024*1024/n)
    SEs = []
    np.random.seed(42)
    if p > n:  # full
        p = n
        pivot_p = np.arange(p)
        p_vis = np.zeros((p, n), dtype=np.int8)
        p_next = np.zeros((p, n), dtype=np.int32)
        p_spd = np.zeros((p, n), dtype=np.float32)
        pivot_spd(edgesrc, edgetgt, pivot_p, p_spd, p_vis, p_next)
        scale = bestScale_np(p_spd, pos, pivot_p)
        res_np = np.zeros(p, dtype=np.float32)
        getSE_np(p_spd, pivot_p, pos * scale, res_np)

        SEp = res_np.sum() / (n * p)
        SE = 0.5 * SEp * n * (n-1)
        SEs.append(SE)
    else:
        for i in range(5):
            pivot_p = np.random.choice(np.arange(n), p, replace=False)
            p_vis = np.zeros((p, n), dtype=np.int8)
            p_next = np.zeros((p, n), dtype=np.int32)
            p_spd = np.zeros((p, n), dtype=np.float32)
            pivot_spd(edgesrc, edgetgt, pivot_p, p_spd, p_vis, p_next)
            scale = bestScale_np(p_spd, pos, pivot_p)
            res_np = np.zeros((p), dtype=np.float32)
            getSE_np(p_spd, pivot_p, pos*scale, res_np)
            SEp = res_np.sum() / (n * p)
            SE = 0.5 * SEp * n * (n-1)
            SEs.append(SE)
    SE = np.array(SEs).mean()
    return SE, 2 * SE / (n) / (n - 1)


@njit
def NP1(pos, edgesrc, edgetgt, kdtree):
    N = len(pos)
    Ns = 0
    cnt = 0
    res = 0.0
    for i in range(N):
        if (N > 1e6 and np.random.random() > N/1e6):
            continue
        nbg = dict()

        for node in edgetgt[edgesrc[i]:edgesrc[i+1]]:
            nbg[node] = 1
        nbglen = len(nbg)
        if (nbglen == 0):
            continue
        Ns = Ns + 1
        _, nbo = kdtree.query(pos[i], nbglen + 1)
        intersect1d = 0
        for node in nbo[0]:
            if node in nbg:
                intersect1d += 1
        res = res + (intersect1d) / (2 * (nbglen) - intersect1d)
    return res / Ns


@njit
def NP2(pos, edgesrc, edgetgt, kdtree):
    N = len(pos)
    Ns = 0
    cnt = 0
    res = 0.0
    res2 = 0.0
    for i in range(N):
        if (N > 1e6 and np.random.random() > N/1e6):
            continue
        nbg = dict()
        for node in edgetgt[edgesrc[i]:edgesrc[i+1]]:
            nbg[node] = 1
            for nodek in edgetgt[edgesrc[node]:edgesrc[node+1]]:
                nbg[nodek] = 1
        nbglen = len(nbg)
        if (nbglen == 0):
            continue
        Ns += 1
        _, nbo = kdtree.query(pos[i], nbglen)
        intersect1d = 0
        for node in nbo[0]:
            if node in nbg:
                intersect1d += 1
        res2 = res2 + (intersect1d - 1) / (2 * (nbglen - 1) - intersect1d + 1)
    return res2/Ns


@njit(parallel=True, cache=True)
def MA(pos, edgesrc, edgetgt):
    N = len(pos)
    Ns = 0
    cnt = 0
    res = 0.0
    for i in prange(N):
        temp = edgesrc[i+1] - edgesrc[i]
        temp_res = 100.0
        for j in range(temp):
            node1 = edgetgt[edgesrc[i]+j]
            for k in range(j):
                node2 = edgetgt[edgesrc[i]+k]
                v1 = pos[node1] - pos[i]
                v2 = pos[node2] - pos[i]
                cos = (v1 * v2).sum() / \
                    np.sqrt((v1**2).sum() * (v2**2).sum() + 1e-32)
                if cos >= 1.0 - 1e-16:
                    cos = 1.0 - 1e-16
                elif cos <= 0.0 + 1e-16:
                    cos = 0.0 + 1e-16
                temparcos = np.arccos(cos)
                temp_res = min(temparcos, temp_res)
        if (temp > 1):
            Ns = Ns + 1
            ideal = 2 * np.pi / temp
            res += (ideal - temp_res) / ideal
    return 1 - res / Ns


@njit(parallel=True, cache=True)
def ECCPU(edgepos):
    res1 = 0.0
    eps = 1e-36
    for i in prange(len(edgepos)//4):
        for j in range(i):
            V10x = edgepos[4*i+2] - edgepos[4*i+0]
            V11x = edgepos[4*j+0] - edgepos[4*i+0]
            V12x = edgepos[4*j+2] - edgepos[4*i+0]
            V10y = edgepos[4*i+3] - edgepos[4*i+1]
            V11y = edgepos[4*j+1] - edgepos[4*i+1]
            V12y = edgepos[4*j+3] - edgepos[4*i+1]

            V10x2 = edgepos[4*j+2] - edgepos[4*j+0]
            V11x2 = edgepos[4*i+0] - edgepos[4*j+0]
            V12x2 = edgepos[4*i+2] - edgepos[4*j+0]
            V10y2 = edgepos[4*j+3] - edgepos[4*j+1]
            V11y2 = edgepos[4*i+1] - edgepos[4*j+1]
            V12y2 = edgepos[4*i+3] - edgepos[4*j+1]
            crs1temp = (V10x*V11y - V10y*V11x)
            crs1temp2 = (V10x*V12y - V10y*V12x)
            crs1 = crs1temp * crs1temp2
            crs1temp = (V10x2*V11y2 - V10y2*V11x2)
            crs1temp2 = (V10x2*V12y2 - V10y2*V12x2)
            crs2 = crs1temp * crs1temp2
#             res1 += ((crs1 <= 0) & (crs2 <= 0))
            res1 += ((crs1 <= 0) & (crs2 <= 0)) - \
                    (((crs1 <= 0) & (crs2 <= 0))
                     & ((crs1temp == 0) & (crs1temp2 == 0) &
                        ((((edgepos[4 * j + 2] - edgepos[4 * i + 2]) * (edgepos[4 * j + 2] - edgepos[4 * i + 0]) > eps * eps)
                          & ((edgepos[4 * j + 0] - edgepos[4 * i + 2]) * (edgepos[4 * j + 0] - edgepos[4 * i + 0]) > eps * eps))
                        | (((edgepos[4 * j + 3] - edgepos[4 * i + 3]) * (edgepos[4 * j + 3] - edgepos[4 * i + 1]) > eps * eps)
                           & ((edgepos[4 * j + 1] - edgepos[4 * i + 3]) * (edgepos[4 * j + 1] - edgepos[4 * i + 1]) > eps * eps))))
                     )
            #  Two line segments parallel
    return res1


ECCu = cnp.RawKernel(""" 
#define eps (1e-16)
extern "C" __global__ void ECCu(const double *edgepos, int *output, int E)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;
    double V10x, V20x, V10y, V20y, crs1, crs2, interx, intery;
    if (i >= E)
        return;
    output[i] = 0;
    for (j = 0; j < i; j++)
    {
        V10x = edgepos[4 * i + 2] - edgepos[4 * i + 0];
        V10y = edgepos[4 * i + 3] - edgepos[4 * i + 1];

        V20x = edgepos[4 * j + 2] - edgepos[4 * j + 0];
        V20y = edgepos[4 * j + 3] - edgepos[4 * j + 1];

        crs1 = (V10x * V20y - V10y * V20x);
        if ((crs1 < eps) & (crs1 > -eps)){ // two line segments parallel 
            output[i] +=( ( ((edgepos[4 * j + 2] - edgepos[4 * i + 2]) * (edgepos[4 * j + 2] - edgepos[4 * i + 0]) < eps * eps) 
            & ((edgepos[4 * j + 3] - edgepos[4 * i + 3]) * (edgepos[4 * j + 3] - edgepos[4 * i + 1]) < eps * eps) )  
            | ( ((edgepos[4 * j + 0] - edgepos[4 * i + 2]) * (edgepos[4 * j + 0] - edgepos[4 * i + 0]) < eps * eps) 
            & ((edgepos[4 * j + 1] - edgepos[4 * i + 3]) * (edgepos[4 * j + 1] - edgepos[4 * i + 1]) < eps * eps) ) ) ;
        }
        else{
            crs2 = ((edgepos[4 * j + 2] - edgepos[4 * i + 2]) * V20y - (edgepos[4 * j + 3] - edgepos[4 * i + 3]) * V20x) / crs1;
            interx = crs2 * V10x + edgepos[4 * i + 2];
            intery = crs2 * V10y + edgepos[4 * i + 3];
            output[i] += ((interx - edgepos[4 * i + 0]) * (interx - edgepos[4 * i + 2]) < eps*eps) 
            & ((interx - edgepos[4 * j + 0]) * (interx - edgepos[4 * j + 2]) < eps*eps);
        }

    }
}
""", "ECCu")


def EC(pos, graph, edgesrc):
    edges = np.hstack([graph.nonzero()[0].reshape(-1, 1),
                      graph.nonzero()[1].reshape(-1, 1)])
    E = len(edges)
    blockdim = (256,)
    ec = 0
    edgengb = 0
    if E < 5e5:
        ec += ECCPU(pos[edges].reshape(-1))
    elif E < 5e6:  # use GPU to calculate
        griddim = ((E - 1)//blockdim[0] + 1, )
        edgeposcu = cnp.array(pos[edges].reshape(-1), dtype=cnp.float64)
        outputcu = cnp.zeros(E, dtype=cnp.int32)
        ECCu(griddim, blockdim, (edgeposcu, outputcu, E))
        ec += outputcu.sum().get()
    else:
        ec += 0
    if E >= 5e6:
        pass
    else:
        temp = 0
        edgengb += ec
        for item in edgesrc:
            nedge = item - temp
            ec -= (nedge * (nedge - 1)) // 2
            temp = item
        edgengb = edgengb - ec
    return int(ec), 1 - np.sqrt(ec/(E*(E-1)/2 - edgengb))
