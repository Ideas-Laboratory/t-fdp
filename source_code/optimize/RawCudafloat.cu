extern "C" __global__ void AttrForce_cu(float *attr_force, float *dC, const float *pos, const int *edgesrc, const int *edgetgt, int N, float a1, float d3alpha)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int xStride = blockDim.x * gridDim.x;
    int cntTgt;
    float bias,mv0, mv1, d, R;
    for (int i = tid; i < N; i += xStride)
    {
        int cntSrc = edgesrc[i + 1] - edgesrc[i];
        attr_force[2 * i] = 0.0;
        attr_force[2 * i + 1] = 0.0;
        for (int k = edgesrc[i]; k < edgesrc[i + 1]; k++)
        {
            int j = edgetgt[k];
            cntTgt = edgesrc[j + 1] - edgesrc[j];
            bias = 1.0f * cntTgt / (cntTgt + cntSrc);
            mv0 = pos[2 * i] + dC[2 * i] - pos[2 * j] - dC[2 * j];
            mv1 = pos[2 * i + 1] + dC[2 * i + 1] - pos[2 * j + 1] - dC[2 * j + 1];
            d = sqrtf(mv0 * mv0 + mv1 * mv1);
            R = d3alpha * (1.0f * a1 * powf(1.0f + d * d, -1.0f) + bias);
            attr_force[2 * i] -= R * mv0;
            attr_force[2 * i + 1] -= R * mv1;
        }
        mv0 = attr_force[2 * i];
        mv1 = attr_force[2 * i+1];
        d = sqrtf(mv0 * mv0 + mv1 * mv1) + 1e-12;
        R = d < 2000.0f ? d : 2000.0f;
        attr_force[2 * i] =  mv0 / d * R;
        attr_force[2 * i + 1] =  mv1 / d * R;
    }
}

extern "C" __global__ void RepulsiveForce_cu(float *dC, const float *pos, int N, float a, float b, float c, float alpha)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int xStride = blockDim.x * gridDim.x;
    float mv0,mv1,dsqare,R;
    for (int i = tid; i < N; i += xStride)
    {
        for (int j = 0; j < N; j++)
        {
            mv0 = pos[2 * i] - pos[2 * j];
            mv1 = pos[2 * i + 1] - pos[2 * j + 1];
            dsqare = (mv0 * mv0 + mv1 * mv1) + 1e-24;
            R = alpha * a * powf(1.0 + dsqare * b, -c);
            dC[2 * i] += R * mv0;
            dC[2 * i + 1] += R * mv1;
        }
    }
}

extern "C" __global__ void Apply_cu(const float *dC, float *pos, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int xStride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += xStride)
    {
        float mv0 = dC[2 * i];
        float mv1 = dC[2 * i + 1];
        float d = sqrtf(mv0 * mv0 + mv1 * mv1) + 1e-16;
        float R = d < 1.0f ? d : 1.0f;
        pos[2 * i] += mv0 / d * R;
        pos[2 * i + 1] += mv1 / d * R;
    }
}

extern "C" __global__ void Y_in_box_cu(float *y_in_box, const float *Y, const int *box_idx, float *box_width, float *min_coord, int n_boxes_per_dim, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int xStride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += xStride)
    {
        int j = box_idx[2 * i];
        int k = box_idx[2 * i + 1];
        y_in_box[2 * i] = (Y[2 * i] - *box_width * j - *min_coord);
        y_in_box[2 * i + 1] = (Y[2 * i + 1] - *box_width * k - *min_coord);
    }
}

extern "C" __global__ void Y_in_box_cu_bak(float *y_in_box, const float *Y, const long long *box_idx, const float *box_lower_bounds, int n_boxes_per_dim, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int xStride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += xStride)
    {
        int j = box_idx[2 * i];
        int k = box_idx[2 * i + 1];
        y_in_box[2 * i] = (Y[2 * i] - box_lower_bounds[2 * (j * n_boxes_per_dim + k)]);
        y_in_box[2 * i + 1] = (Y[2 * i + 1] - box_lower_bounds[2 * (j * n_boxes_per_dim + k) + 1]);
    }
}

extern "C" __global__ void Interpolate_cu(const float *y_in_box, const float *y_tilde_spacings, const float *denominator, float *interpolated_values, int n_interpolation_points, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int xStride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += xStride)
    {
        for (int j = 0; j < n_interpolation_points; j++)
        {
            interpolated_values[2 * (i * n_interpolation_points + j)] = 1.0;
            interpolated_values[2 * (i * n_interpolation_points + j) + 1] = 1.0;
            for (int k = 0; k < n_interpolation_points; k++)
            {
                if (j != k)
                {
                    interpolated_values[2 * (i * n_interpolation_points + j)] *= y_in_box[2 * i] - y_tilde_spacings[k];
                    interpolated_values[2 * (i * n_interpolation_points + j) + 1] *= y_in_box[2 * i + 1] - y_tilde_spacings[k];
                }
            }
            interpolated_values[2 * (i * n_interpolation_points + j)] /= denominator[j];
            interpolated_values[2 * (i * n_interpolation_points + j) + 1] /= denominator[j];
        }
    }
}

extern "C" __global__ void Denominator_cu(float *denominator, const float *y_tilde_spacings, int n_interpolation_points)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    denominator[tid] = 1;
    for (int j = 0; j < n_interpolation_points; j++)
    {
        if (tid != j)
            denominator[tid] *= y_tilde_spacings[tid] - y_tilde_spacings[j];
    }
}

extern "C" __global__ void Compute_w_coeff_cu(float *w_coefficients, const int *box_idx, const float *chargesQij, const float *interpolated_values, int n_interpolation_points, int n_boxes_per_dim, int n_terms, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int xStride = blockDim.x * gridDim.x;
    int w_coeff_len_per_dim = n_interpolation_points * n_boxes_per_dim;
    for (int i = tid; i < N; i += xStride)
    {
        for (int j = 0; j < n_interpolation_points; j++)
        {
            for (int k = 0; k < n_interpolation_points; k++)
            {
                int idj = box_idx[2 * i];
                int idk = box_idx[2 * i + 1];
                int point = (idj * n_interpolation_points + j) * w_coeff_len_per_dim + idk * n_interpolation_points + k;
                float prob = interpolated_values[2 * (i * n_interpolation_points + j)] * interpolated_values[2 * (i * n_interpolation_points + k) + 1];
                for (int term = 0; term < n_terms; term++)
                {
                    atomicAdd(&(w_coefficients[n_terms * point + term]), prob * chargesQij[n_terms * i + term]);
                }
            }
        }
    }
}

extern "C" __global__ void PotentialsQij_cu(float *PotentialsQij, const int *box_idx, const float *interpolated_values, const float *y_tilde_values, int n_interpolation_points, int n_boxes_per_dim, int n_terms, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int xStride = blockDim.x * gridDim.x;
    int Qij_len_per_dim = n_interpolation_points * n_boxes_per_dim;

    for (int i = tid; i < N; i += xStride)
    {
        for (int j = 0; j < n_interpolation_points; j++)
        {
            for (int k = 0; k < n_interpolation_points; k++)
            {
                int idj = box_idx[2 * i];
                int idk = box_idx[2 * i + 1];
                int point = (idj * n_interpolation_points + j + Qij_len_per_dim) * 2 * Qij_len_per_dim + Qij_len_per_dim + idk * n_interpolation_points + k;
                float prob = interpolated_values[2 * (i * n_interpolation_points + j)] * interpolated_values[2 * (i * n_interpolation_points + k) + 1];
                for (int term = 0; term < n_terms; term++)
                {
                    PotentialsQij[i * n_terms + term] += prob * y_tilde_values[term * 4 * Qij_len_per_dim * Qij_len_per_dim + point];
                }
            }
        }
    }
}

extern "C" __global__ void Circulant_kernel_tilde_cu(float *circulant_kernel_tilde, float *box_width, int n_interpolation_points, int n_interpolation_points_1d, int n_fft_coeffs, int n_boxes_per_dim, int N, float a, float c)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int xStride = blockDim.x * gridDim.x;
    float whsquare = *box_width / n_interpolation_points;
    float tmp;
    whsquare = whsquare * whsquare;
    
    for (int i = tid; i < n_interpolation_points_1d; i += xStride)
    {
        for (int j = 0; j < n_interpolation_points_1d; j++)
        {
            tmp = a * powf(1.0f + (i * i + j * j) * whsquare, -c);
            circulant_kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
            circulant_kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
            circulant_kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
            circulant_kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
        }
        
    }
}

extern "C" __global__ void Collision_kernel_tilde_cu(float *circulant_kernel_tilde, float *box_width, int n_interpolation_points, int n_interpolation_points_1d, int n_fft_coeffs, int n_boxes_per_dim, int N, float Collision_size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int xStride = blockDim.x * gridDim.x;
    float whsquare = *box_width / n_interpolation_points;
    whsquare *= whsquare;

    for (int i = tid; i < n_interpolation_points_1d; i += xStride)
    {
        for (int j = 0; j < n_interpolation_points_1d; j++)
        {
            // float tmp = powf(1.0 + (i * i + j * j) * whsquare, -c);
            float tmp = sqrtf((i * i + j * j) * whsquare) + 1e-3;
            tmp = max(Collision_size - tmp,0.0) / tmp;
            circulant_kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
            circulant_kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
            circulant_kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
            circulant_kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
        }
    }
}

extern "C" __global__ void Box_idx_cu(int *box_idx, const float *Y, float *box_width, float *min_coord, int n_boxes_per_dim, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int xStride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += xStride)
    {
        box_idx[2 * i] = max(0, min(int((Y[2 * i] - *min_coord) / (*box_width)), n_boxes_per_dim - 1));
        box_idx[2 * i + 1] = max(0, min(int((Y[2 * i + 1] - *min_coord) / (*box_width)), n_boxes_per_dim - 1));
    }
}
