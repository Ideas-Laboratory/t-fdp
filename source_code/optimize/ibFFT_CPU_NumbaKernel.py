from numba import njit, prange, jit
import numpy as np


@njit(parallel=False, cache=True)
def Circulant_kernel_tilde(circulant_kernel_tilde, box_width, n_interpolation_points, n_interpolation_points_1d,  n_fft_coeffs, n_boxes_per_dim, N, a, c):
    whsquare = box_width / n_interpolation_points
    whsquare *= whsquare
    for i in prange(n_interpolation_points_1d):
        for j in range(n_interpolation_points_1d):
            tmp = a * np.power(1.0 + (i * i + j * j) * whsquare, -c)
            circulant_kernel_tilde[(
                n_interpolation_points_1d + i)][(n_interpolation_points_1d + j)] = tmp
            circulant_kernel_tilde[(
                n_interpolation_points_1d - i)][(n_interpolation_points_1d + j)] = tmp
            circulant_kernel_tilde[(
                n_interpolation_points_1d + i)][(n_interpolation_points_1d - j)] = tmp
            circulant_kernel_tilde[(
                n_interpolation_points_1d - i)][(n_interpolation_points_1d - j)] = tmp


@njit(parallel=False, cache=True)
def Box_idx(box_idx, Y, box_width, min_coord, n_boxes_per_dim, N):
    for i in prange(N):
        box_idx[i][0] = max(
            0, min(int((Y[i][0] - min_coord) / (box_width)), n_boxes_per_dim - 1))
        box_idx[i][1] = max(
            0, min(int((Y[i][1] - min_coord) / (box_width)), n_boxes_per_dim - 1))


@njit(parallel=False, cache=True)
def Y_in_box(y_in_box, Y, box_idx, box_width, min_coord, n_boxes_per_dim, N):
    for i in prange(N):
        j = box_idx[i][0]
        k = box_idx[i][1]
        y_in_box[i][0] = (Y[i][0] - box_width * j - min_coord)
        y_in_box[i][1] = (Y[i][1] - box_width * k - min_coord)


@njit(parallel=False, cache=True)
def Interpolate(y_in_box, y_tilde_spacings, denominator, interpolated_values, n_interpolation_points, N):
    for i in prange(N):
        for j in range(n_interpolation_points):
            interpolated_values[i][j][0] = 1.0
            interpolated_values[i][j][1] = 1.0
            for k in range(n_interpolation_points):
                if j != k:
                    interpolated_values[i][j][0] *= y_in_box[i][0] - \
                        y_tilde_spacings[k]
                    interpolated_values[i][j][1] *= y_in_box[i][1] - \
                        y_tilde_spacings[k]
            interpolated_values[i][j][0] /= denominator[j]
            interpolated_values[i][j][1] /= denominator[j]


@njit(parallel=False)
def Compute_w_coeff(w_coefficients, box_idx, chargesQij, interpolated_values, n_interpolation_points, n_boxes_per_dim, n_terms, N):
    for i in range(N):
        for j in range(n_interpolation_points):
            for k in range(n_interpolation_points):
                idj = box_idx[i][0] * n_interpolation_points + j
                idk = box_idx[i][1] * n_interpolation_points + k
                prob = interpolated_values[i][j][0] * \
                    interpolated_values[i][k][1]
                for term in range(n_terms):
                    w_coefficients[idj][idk][term] += prob * \
                        chargesQij[i][term]


@njit(parallel=False)
def PotentialsQij(potentialsQij, box_idx, interpolated_values, y_tilde_values, n_interpolation_points, n_boxes_per_dim, n_terms, N):
    Qij_len_per_dim = n_interpolation_points * n_boxes_per_dim
    for i in prange(N):
        for j in range(n_interpolation_points):
            for k in range(n_interpolation_points):
                idj = box_idx[i][0] * \
                    n_interpolation_points + j + Qij_len_per_dim
                idk = box_idx[i][1] * \
                    n_interpolation_points + k + Qij_len_per_dim
                prob = interpolated_values[i][j][0] * \
                    interpolated_values[i][k][1]
                for term in range(n_terms):
                    potentialsQij[i][term] += prob * \
                        y_tilde_values[term][idj][idk]


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
def ReplForce(dC, pos, N, a, b, c, alpha):
    for i in prange(N):
        for j in range(N):
            mv0 = pos[i][0] - pos[j][0]
            mv1 = pos[i][1] - pos[j][1]
            dsqare = mv0 * mv0 + mv1 * mv1
            R = alpha * a * np.power(1.0 + dsqare * b, -c)
            dC[i][0] += R * mv0
            dC[i][1] += R * mv1
