import numpy as np
import sys
import cupy as cnp
import time
import os
grid_dim = (256,)
block_dim = (512,)
cudarawtext = open(os.path.join(
    os.path.dirname(__file__), "RawCudafloat.cu")).read()
AttrForce_cu = cnp.RawKernel(cudarawtext, "AttrForce_cu")
RepulsiveForce_cu = cnp.RawKernel(cudarawtext, "RepulsiveForce_cu")
ApplyForce_cu = cnp.RawKernel(cudarawtext, "Apply_cu")
Y_in_box_cu = cnp.RawKernel(cudarawtext, "Y_in_box_cu")
Y_in_box_cu_bak = cnp.RawKernel(cudarawtext, "Y_in_box_cu_bak")
Circulant_kernel_tilde_cu = cnp.RawKernel(
    cudarawtext, "Circulant_kernel_tilde_cu")
Collision_kernel_tilde_cu = cnp.RawKernel(
    cudarawtext, "Collision_kernel_tilde_cu")
Interpolate_cu = cnp.RawKernel(cudarawtext, "Interpolate_cu")
Denominator_cu = cnp.RawKernel(cudarawtext, "Denominator_cu")
Compute_w_coeff_cu = cnp.RawKernel(cudarawtext, "Compute_w_coeff_cu")

PotentialsQij_cu = cnp.RawKernel(cudarawtext, "PotentialsQij_cu")
Circulant_kernel_tilde_cu = cnp.RawKernel(
    cudarawtext, "Circulant_kernel_tilde_cu")
Box_idx_cu = cnp.RawKernel(cudarawtext, "Box_idx_cu")


def ibFFT_repulsive(Y, n_interpolation_points, intervals_per_integer, min_num_intervals, gamma, paraFactor):
    max_coord = Y.max()
    min_coord = Y.min()
    N = Y.shape[0]
    n_boxes_per_dim = int(np.minimum(np.sqrt(16*N), np.maximum(np.sqrt(8 * N/np.log(N)), np.maximum(
        min_num_intervals, (max_coord.item() - min_coord.item()) / intervals_per_integer))))
    allowed_n_boxes_per_dim = np.array([50, 54,   64,   72,   81, 96,  108,  128,  144,  162,  192,  216,  243,
                                        256,  288,  324,   384,  432, 576, 648,  768,  864,  972, 1152, 1296])
    if (n_boxes_per_dim < allowed_n_boxes_per_dim[-1]):
        n_boxes_per_dim = allowed_n_boxes_per_dim[(
            allowed_n_boxes_per_dim > n_boxes_per_dim)][0]
    else:
        n_boxes_per_dim = allowed_n_boxes_per_dim[-1]
    n_boxes_per_dim = n_boxes_per_dim.item()
    # print("n_boxes_per_dim",n_boxes_per_dim)
    squared_n_terms = 3
    n_terms = squared_n_terms
    ChargesQij = cnp.ones((N, squared_n_terms), dtype=cnp.float32)
    ChargesQij[:, :2] = Y

    box_width = (max_coord - min_coord)/n_boxes_per_dim
    n_boxes = n_boxes_per_dim * n_boxes_per_dim
    n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim
    n_fft_coeffs = 2 * n_interpolation_points_1d

    h = 1.0 / n_interpolation_points
    y_tilde_spacings = cnp.array(cnp.arange(
        n_interpolation_points) * h + h/2, dtype=cnp.float32)
    circulant_kernel_tilde = cnp.zeros(
        (n_fft_coeffs, n_fft_coeffs), dtype=cnp.float32)
    Circulant_kernel_tilde_cu(grid_dim, block_dim, (circulant_kernel_tilde, box_width, n_interpolation_points, n_interpolation_points_1d,
                              n_fft_coeffs, n_boxes_per_dim, N, cnp.float32(paraFactor), cnp.float32(gamma)))

    fft_kernel_tilde = cnp.fft.rfft2(circulant_kernel_tilde)
    box_idx = cnp.ndarray((N, 2), dtype=cnp.int32)

    Box_idx_cu(grid_dim, block_dim, (box_idx, Y,
               box_width, min_coord, n_boxes_per_dim, N))
    y_in_box = cnp.zeros_like(Y)
    Y_in_box_cu(grid_dim, block_dim, (y_in_box, Y, box_idx,
                box_width, min_coord, n_boxes_per_dim, N))

    denominator = cnp.ndarray(n_interpolation_points, dtype=cnp.float32)
    Denominator_cu((1,), (n_interpolation_points,),
                   (denominator, y_tilde_spacings, n_interpolation_points))

    interpolate_values = cnp.ndarray(
        (N, n_interpolation_points, 2), dtype=cnp.float32)
    Interpolate_cu(grid_dim, block_dim, (y_in_box, y_tilde_spacings,
                   denominator, interpolate_values, n_interpolation_points, len(Y)))
    w_coefficients = cnp.zeros((n_boxes_per_dim * n_interpolation_points,
                               n_boxes_per_dim * n_interpolation_points, squared_n_terms), dtype=cnp.float32)
    Compute_w_coeff_cu(grid_dim, block_dim, (w_coefficients, box_idx, ChargesQij,
                       interpolate_values, n_interpolation_points, n_boxes_per_dim, n_terms, N))
    mat_w = cnp.zeros((2*n_boxes_per_dim * n_interpolation_points, 2 *
                      n_boxes_per_dim * n_interpolation_points, n_terms), dtype=cnp.float32)
    mat_w[:n_boxes_per_dim * n_interpolation_points,
          :n_boxes_per_dim * n_interpolation_points] = w_coefficients
    mat_w = mat_w.transpose((2, 0, 1))
    fft_w = cnp.fft.rfft2(mat_w)
    rmut = fft_w * fft_kernel_tilde
    output = cnp.fft.irfft2(rmut)

    PotentialsQij = cnp.zeros((N, n_terms), dtype=cnp.float32)
    PotentialsQij_cu(grid_dim, block_dim, (PotentialsQij, box_idx, interpolate_values,
                     output, n_interpolation_points, n_boxes_per_dim, n_terms, N))

    neg_f = cnp.ndarray((N, 2), dtype=cnp.float32)
    PotentialsCom = PotentialsQij[:, 2].reshape((-1, 1))
    PotentialsXY = PotentialsQij[:, :2]
    neg_f = PotentialsCom * Y - PotentialsXY

    return neg_f


def ibFFT_GPU(pos, edgesrc, edgetgt, n_interpolation_points=3, intervals_per_integer=1.0, min_num_intervals=100,
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
    pos = cnp.array(pos, dtype=cnp.float32)
    st = time.time()
    dC = cnp.zeros((N, 2), dtype=cnp.float32)
    attr_force = cnp.zeros((N, 2), dtype=cnp.float32)
    pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min())/2
    pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min())/2
    edgesrc = cnp.array(edgesrc, dtype=cnp.int32)
    edgetgt = cnp.array(edgetgt, dtype=cnp.int32)
    if seed is not None:
        cnp.random.seed(seed)
    for it in range(max_iter):
        if combine:
            if it == (18*max_iter//20):
                n_interpolation_points = 2
            if it == (19*max_iter//20):
                n_interpolation_points = 3
        AttrForce_cu(grid_dim, block_dim, (attr_force, dC, pos, edgesrc,
                     edgetgt, N, cnp.float32(beta), cnp.float32(d3alpha)))
        dC += attr_force
        dC -= 0.01 * d3alpha * \
            cnp.random.randn(pos.shape[0], pos.shape[1], dtype=cnp.float32)
        dC += d3alpha * ibFFT_repulsive(pos, n_interpolation_points,
                                        intervals_per_integer, min_num_intervals, cnp.float32(gamma), cnp.float32(paraFactor))
        ApplyForce_cu(grid_dim, block_dim, (dC, pos, N))
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
    return pos.get(), ed-st
