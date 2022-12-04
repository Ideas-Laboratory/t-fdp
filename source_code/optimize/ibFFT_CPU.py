import numpy as np
import sys
import torch
import time
from .ibFFT_CPU_NumbaKernel import *
from pyfftw.interfaces.numpy_fft import rfft2, irfft2
import pyfftw
# import numba
pyfftw.interfaces.cache.enable()
# pyfftw.config.NUM_THREADS = 8
# numba.set_num_threads(8)


def ibFFT_repulsive(Y, n_interpolation_points, intervals_per_integer, min_num_intervals, gamma, paraFactor):
    max_coord = Y.max()
    min_coord = Y.min()
    N = Y.shape[0]
    n_boxes_per_dim = int(np.minimum(np.sqrt(16*N), np.maximum(np.sqrt(4 * N/np.log(N)), np.maximum(
        min_num_intervals, (max_coord.item() - min_coord.item()) / intervals_per_integer))))
    allowed_n_boxes_per_dim = np.array([50, 54,   64,   72,   81, 96,  108,  128,  144,  162,  192,  216,  243,
                                        256,  288,  324,   384,  432, 576, 648,  768,  864,  972, 1152, 1296])
    if (n_boxes_per_dim < allowed_n_boxes_per_dim[-1]):
        n_boxes_per_dim = allowed_n_boxes_per_dim[(
            allowed_n_boxes_per_dim > n_boxes_per_dim)][0]
    else:
        n_boxes_per_dim = allowed_n_boxes_per_dim[-1]
    n_boxes_per_dim = n_boxes_per_dim.item()
    squared_n_terms = 3
    n_terms = squared_n_terms
    ChargesQij = np.ones((N, squared_n_terms), dtype=np.float32)
    ChargesQij[:, :2] = Y

    box_width = (max_coord - min_coord)/n_boxes_per_dim
    n_boxes = n_boxes_per_dim * n_boxes_per_dim
    n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim
    n_fft_coeffs = 2 * n_interpolation_points_1d

    whsquare = box_width / n_interpolation_points
    whsquare *= whsquare
    h = 1.0 / n_interpolation_points * box_width
    y_tilde_spacings = np.array(
        np.arange(n_interpolation_points) * h + h/2, dtype=np.float32)

    if int(gamma) == gamma:
        half_kernel = paraFactor / ((1.0 + (np.arange(n_interpolation_points_1d)**2 + (
            np.arange(n_interpolation_points_1d)**2).reshape(-1, 1)) * whsquare)**int(gamma))
    else:
        half_kernel = paraFactor * np.power(1.0 + (np.arange(n_interpolation_points_1d)**2 + (
            np.arange(n_interpolation_points_1d)**2).reshape(-1, 1)) * whsquare, -gamma)
    circulant_kernel_tilde = np.zeros(
        (n_fft_coeffs, n_fft_coeffs), dtype=np.float32)
    circulant_kernel_tilde[n_interpolation_points_1d:,
                           n_interpolation_points_1d:] = half_kernel
    circulant_kernel_tilde[1:n_interpolation_points_1d+1,
                           n_interpolation_points_1d:] = np.flipud(half_kernel)
    circulant_kernel_tilde[n_interpolation_points_1d:,
                           1:n_interpolation_points_1d+1] = np.fliplr(half_kernel)
    circulant_kernel_tilde[1:n_interpolation_points_1d+1,
                           1:n_interpolation_points_1d+1] = np.fliplr(np.flipud(half_kernel))
    fft_kernel_tilde = rfft2(circulant_kernel_tilde)
    # fft_kernel_tilde = rfft2(circulant_kernel_tilde,threads=4)
    box_idx = np.ndarray((N, 2), dtype=np.int32)

    Box_idx(box_idx, Y, box_width, min_coord, n_boxes_per_dim, N)

    y_in_box = np.zeros_like(Y)

    Y_in_box(y_in_box, Y, box_idx, box_width, min_coord, n_boxes_per_dim, N)

    denominator_sub = (y_tilde_spacings.reshape(-1, 1) - y_tilde_spacings)
    np.fill_diagonal(denominator_sub, 1)
    denominator = denominator_sub.prod(axis=0)

    interpolate_values = np.ndarray(
        (N, n_interpolation_points, 2), dtype=np.float32)
    Interpolate(y_in_box, y_tilde_spacings, denominator,
                interpolate_values, n_interpolation_points, N)

    w_coefficients = np.zeros((n_boxes_per_dim * n_interpolation_points,
                              n_boxes_per_dim * n_interpolation_points, squared_n_terms), dtype=np.float32)

    Compute_w_coeff(w_coefficients, box_idx, ChargesQij, interpolate_values,
                    n_interpolation_points, n_boxes_per_dim, n_terms, N)

    mat_w = np.zeros((2*n_boxes_per_dim * n_interpolation_points, 2 *
                     n_boxes_per_dim * n_interpolation_points, n_terms), dtype=np.float32)
    mat_w[:n_boxes_per_dim * n_interpolation_points,
          :n_boxes_per_dim * n_interpolation_points] = w_coefficients
    mat_w = mat_w.transpose((2, 0, 1))
    # fft_w = rfft2(mat_w,threads=4)
    fft_w = rfft2(mat_w)
    rmut = fft_w * fft_kernel_tilde
    # output = irfft2(rmut,threads=4)
    output = irfft2(rmut)
    potentialsQij = np.zeros((N, n_terms), dtype=np.float32)
    PotentialsQij(potentialsQij, box_idx, interpolate_values,
                  output, n_interpolation_points, n_boxes_per_dim, n_terms, N)
    neg_f = np.ndarray((N, 2), dtype=np.float32)
    PotentialsCom = potentialsQij[:, 2].reshape((-1, 1))
    PotentialsXY = potentialsQij[:, :2]
    neg_f = PotentialsCom * Y - PotentialsXY
    return neg_f


def ibFFT_CPU(pos, edgesrc, edgetgt, n_interpolation_points=3, intervals_per_integer=1.0, min_num_intervals=100,
              alpha=0.1, beta=8, gamma=2, max_iter=300, combine=True, seed=None):
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
    pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min())/2
    pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min())/2
    edgesrc = np.array(edgesrc, dtype=np.int32)
    edgetgt = np.array(edgetgt, dtype=np.int32)
    bias = np.zeros(edgetgt.shape[0], dtype=np.float32)
    computebias(N, edgesrc, edgetgt, bias)
    if seed is not None:
        torch.manual_seed(seed)
    for it in range(max_iter):
        if combine:
            if it == (18*max_iter//20):
                n_interpolation_points = 2
            if it == (19*max_iter//20):
                n_interpolation_points = 3
        AttrForce(attr_force, dC, pos, edgesrc, edgetgt, bias, N, np.float32(
            beta), d3alpha)
        dC += attr_force
        dC -= 0.01 * d3alpha * \
            torch.normal(0, 1, size=pos.shape).numpy().astype(np.float32)
        dC += d3alpha * ibFFT_repulsive(pos, n_interpolation_points,
                                        intervals_per_integer, min_num_intervals, np.float32(gamma), np.float32(paraFactor))
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
