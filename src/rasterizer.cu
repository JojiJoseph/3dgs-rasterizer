#include <torch/extension.h>
#include <cub/cub.cuh>

__device__ void matmul(float *A, float *B, float *C, int m, int n, int k)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            float sum = 0;
            for (int l = 0; l < n; l++)
            {
                sum += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

__device__ void transpose(float *A, float *B, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            B[j * m + i] = A[i * n + j];
        }
    }
}

__device__ float *calc_cov3d(float *scale, float *quat)
{
    float q0 = quat[0];
    float q1 = quat[1];
    float q2 = quat[2];
    float q3 = quat[3];
    float q0_2 = q0 * q0;
    float q1_2 = q1 * q1;
    float q2_2 = q2 * q2;
    float q3_2 = q3 * q3;
    float R[3][3] = {
        {1 - 2 * q2_2 - 2 * q3_2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2},
        {2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1_2 - 2 * q3_2, 2 * q2 * q3 - 2 * q0 * q1},
        {2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1_2 - 2 * q2_2}};
    float S[3][3] = {
        {scale[0], 0, 0},
        {0, scale[1], 0},
        {0, 0, scale[2]}};

    float RS[3][3];
    matmul((float *)R, (float *)S, (float *)RS, 3, 3, 3);
    float RST[3][3];
    transpose((float *)RS, (float *)RST, 3, 3);
    float *cov3d = new float[9];
    matmul((float *)RS, (float *)RST, (float *)cov3d, 3, 3, 3);
    return cov3d;
}

__device__ float calc_radius(float cov[2][2])
{
    float det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    float mid = 0.5f * (cov[0][0] + cov[1][1]);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
    return radius;
}

struct Rect
{
    int x0, y0, x1, y1;
};

__device__ Rect get_rect(int x, int y, int radius, int width, int height)
{
    int x0 = min((width + 15) / 16, max(0, (int)((x - radius) / 16)));
    int x1 = min((width + 15) / 16, max(0, (int)((x + radius + 15) / 16)));
    int y0 = min((height + 15) / 16, max(0, (int)((y - radius) / 16)));
    int y1 = min((height + 15) / 16, max(0, (int)((y + radius + 15) / 16)));
    return {x0, y0, x1, y1};
}

__global__ void preprocess_kernel(float *means3d, float *scales, float *quats, float *viewmat, float *K, float *means2d, bool *valid_gaussian, float *cov_inv, int *radii, int *ntiles, float *depths, int N, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
    {
        return;
    }
    float x = means3d[idx * 3];
    float y = means3d[idx * 3 + 1];
    float z = means3d[idx * 3 + 2];
    float x2 = viewmat[0] * x + viewmat[1] * y + viewmat[2] * z + viewmat[3];
    float y2 = viewmat[4] * x + viewmat[5] * y + viewmat[6] * z + viewmat[7];
    float z2 = viewmat[8] * x + viewmat[9] * y + viewmat[10] * z + viewmat[11];
    float x3 = K[0] * x2 + K[1] * y2 + K[2] * z2;
    float y3 = K[3] * x2 + K[4] * y2 + K[5] * z2;
    means2d[idx * 2] = x3 / z2;
    means2d[idx * 2 + 1] = y3 / z2;
    valid_gaussian[idx] = (z2 > 0.1) && means2d[idx * 2] >= -100 && means2d[idx * 2] < (width + 100) && means2d[idx * 2 + 1] >= -100 && means2d[idx * 2 + 1] < (height + 100);

    float fx = K[0];
    float fy = K[4];

    float scale[3] = {scales[idx * 3 + 0], scales[idx * 3 + 1], scales[idx * 3 + 2]};
    float quat[4] = {quats[idx * 4], quats[idx * 4 + 1], quats[idx * 4 + 2], quats[idx * 4 + 3]};
    float *cov3D = calc_cov3d(scale, quat);

    float J[2][3] = {
        {fx / z2, 0, -fx * x2 / (z2 * z2)},
        {0, fy / z2, -fy * y2 / (z2 * z2)}};

    float W[3][3] = {
        {viewmat[0], viewmat[1], viewmat[2]},
        {viewmat[4], viewmat[5], viewmat[6]},
        {viewmat[8], viewmat[9], viewmat[10]}};

    float WT[3][3];
    transpose((float *)W, (float *)WT, 3, 3);
    float JT[3][2];
    transpose((float *)J, (float *)JT, 2, 3);

    float WJT[3][2];
    matmul((float *)WT, (float *)JT, (float *)WJT, 3, 3, 2);
    float JW[2][3];
    matmul((float *)J, (float *)W, (float *)JW, 2, 3, 3);

    float JWCov[2][3];
    matmul((float *)JW, (float *)cov3D, (float *)JWCov, 2, 3, 3);

    float cov[2][2];
    matmul((float *)JWCov, (float *)WJT, (float *)cov, 2, 3, 2);
    delete cov3D;

    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;

    float det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    float cov_inv_current[2][2] = {
        {cov[1][1] / det, -cov[0][1] / det},
        {-cov[1][0] / det, cov[0][0] / det}};

    cov_inv[idx * 4] = cov_inv_current[0][0];
    cov_inv[idx * 4 + 1] = cov_inv_current[0][1];
    cov_inv[idx * 4 + 2] = cov_inv_current[1][0];
    cov_inv[idx * 4 + 3] = cov_inv_current[1][1];
    float radius = calc_radius(cov);

    radii[idx] = radius;

    Rect rect = get_rect(means2d[idx * 2], means2d[idx * 2 + 1], radius, width, height);
    int number_of_tiles_touched = (rect.x1 - rect.x0) * (rect.y1 - rect.y0);
    ntiles[idx] = number_of_tiles_touched;
    depths[idx] = z2;

}



__global__ void render_kernel_tiled(float *means2d, float *scales, float *opacities, float *color, bool *valid_gaussian, float *cov_inv, int *radii, float *out, uint32_t *tiles_range_start, uint32_t *tiles_range_end, uint64_t *indices_sorted, int width, int height, int N)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    float channel[3] = {0.0, 0.0, 0.0};
    int tile_x = x / 16;
    int tile_y = y / 16;
    int tile_id = tile_y * (width + 15) / 16 + tile_x;
    int offset = tiles_range_start[tile_id];
    int end = tiles_range_end[tile_id];


    __shared__ int tile_indices[256];
    __shared__ float tile_means2d[256 * 2];
    __shared__ float tile_cov_inv[256 * 4];
    __shared__ int tile_radii[256];
    __shared__ float tile_color[256 * 3];
    __shared__ float tile_opacities[256];
    __shared__ bool tile_valid_gaussian[256];
    float T = 1.0;
    for (int i = offset; i < end; i += 256)
    {
        int sorted_idx = i + threadIdx.x + threadIdx.y * 16;
        if (sorted_idx > end)
        {
            continue;
        }
        int idx = indices_sorted[sorted_idx];
        // printf("idx: %d\n", idx);
        tile_indices[threadIdx.x + threadIdx.y * 16] = idx;
        tile_means2d[(threadIdx.x + threadIdx.y * 16) * 2] = means2d[idx * 2];
        tile_means2d[(threadIdx.x + threadIdx.y * 16) * 2 + 1] = means2d[idx * 2 + 1];
        tile_cov_inv[(threadIdx.x + threadIdx.y * 16) * 4] = cov_inv[idx * 4];
        tile_cov_inv[(threadIdx.x + threadIdx.y * 16) * 4 + 1] = cov_inv[idx * 4 + 1];
        tile_cov_inv[(threadIdx.x + threadIdx.y * 16) * 4 + 2] = cov_inv[idx * 4 + 2];
        tile_cov_inv[(threadIdx.x + threadIdx.y * 16) * 4 + 3] = cov_inv[idx * 4 + 3];
        tile_radii[threadIdx.x + threadIdx.y * 16] = radii[idx];
        tile_color[(threadIdx.x + threadIdx.y * 16) * 3] = color[idx * 3];
        tile_color[(threadIdx.x + threadIdx.y * 16) * 3 + 1] = color[idx * 3 + 1];
        tile_color[(threadIdx.x + threadIdx.y * 16) * 3 + 2] = color[idx * 3 + 2];
        tile_opacities[threadIdx.x + threadIdx.y * 16] = opacities[idx];
        tile_valid_gaussian[threadIdx.x + threadIdx.y * 16] = valid_gaussian[idx];
        __syncthreads();
        for (int j = 0; j < 256; ++j)
        {
            if (i + j >= end)
            {
                break;
            }
            if (!tile_valid_gaussian[j])
            {
                continue;
            }
            int x0 = tile_means2d[j * 2];
            int y0 = tile_means2d[j * 2 + 1];
            float delta_x = x - x0;
            float delta_y = y - y0;
            float aa = tile_cov_inv[j * 4 + 0];
            float bb = tile_cov_inv[j * 4 + 1];
            float cc = tile_cov_inv[j * 4 + 2];
            float dd = tile_cov_inv[j * 4 + 3];
            float alpha = tile_opacities[j] * exp(-0.5 * (aa * delta_x * delta_x + bb * delta_x * delta_y + cc * delta_y * delta_x + dd * delta_y * delta_y));
            alpha = min(0.99f, alpha);
            alpha = max(0.00f, alpha);
            channel[0] += T * alpha * tile_color[j * 3 + 0];
            channel[1] += T * alpha * tile_color[j * 3 + 1];
            channel[2] += T * alpha * tile_color[j * 3 + 2];
            // channel[2] = 1.0;//tile_color[j*3 + 2];
            T = T * (1 - alpha);
            if (T < 0.001)
            {
                break;
            }
        }
    }
    
    out[(y * width + x) * 3 + 0] = channel[0];
    out[(y * width + x) * 3 + 1] = channel[1];
    out[(y * width + x) * 3 + 2] = channel[2];
}

__global__ void duplicate_and_assign_key_kernel(
    uint32_t *offset,
    float *means2d,
    float *depths,
    float *viewmat,
    int *radii,
    float *K,
    int N,
    uint64_t *keys,   // Out
    uint64_t *indices // Out
)
{
    uint32_t total_indices = offset[N - 1];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
    {
        return;
    }
    if (radii[idx] <= 0)
    {
        return;
    }

    int width = K[2] * 2;
    int height = K[5] * 2;
    Rect rect = get_rect(means2d[idx * 2], means2d[idx * 2 + 1], radii[idx], width, height);
    int off = (idx == 0 ? 0 : offset[idx - 1]);
    for (int i = rect.y0; i < rect.y1; ++i)
    {
        for (int j = rect.x0; j < rect.x1; ++j)
        {
            uint64_t key = i * (width + 15) / 16 + j;
            key = key << 32;
            key = key | *((uint32_t *)(&depths[idx]));
            keys[off] = key;
            indices[off] = idx;
            off++;

        }
    }
}

__global__ void identify_tiles_kernel(
    uint64_t *keys_sorted,
    uint64_t *indices_sorted,
    uint32_t total_keys,
    uint32_t *tiles_range_start,
    uint32_t *tiles_range_end,
    int width,
    int height)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_keys)
    {
        return;
    }
    int tile = keys_sorted[idx] >> 32;
    if (idx == 0)
    {
        tiles_range_start[0] = 0;
    }
    else
    {
        int prev_tile = keys_sorted[idx - 1] >> 32;
        if (tile != prev_tile)
        {
            tiles_range_start[tile] = idx;
            tiles_range_end[prev_tile] = idx;
        }
    }
    if (idx == total_keys - 1)
    {
        tiles_range_end[tile] = total_keys;
    }
}

torch::Tensor render(torch::Tensor means3d, torch::Tensor quats, torch::Tensor colors, torch::Tensor opacities, torch::Tensor scales, torch::Tensor viewmat, torch::Tensor K, int width, int height)
{
    int N = means3d.size(0);
    // int threads = 1024;
    // int blocks = (N + threads - 1) / threads;
    int threads = 16;
    dim3 blocks((width + threads - 1) / threads, (height + threads - 1) / threads);
    dim3 threads2(threads, threads);

    torch::Tensor means2d = torch::zeros({N, 2}).cuda();
    torch::Tensor valid_gaussian = torch::zeros({N}, torch::kBool).cuda();
    torch::Tensor cov_inv = torch::zeros({N, 4}).cuda();
    torch::Tensor radii = torch::zeros({N}, torch::kInt).cuda();
    torch::Tensor ntiles = torch::zeros({N}, torch::kInt).cuda();
    torch::Tensor depths = torch::zeros({N}).cuda();

    preprocess_kernel<<<(N + threads - 1) / threads, threads>>>(
        means3d.data_ptr<float>(),
        scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        viewmat.data_ptr<float>(),
        K.data_ptr<float>(),
        means2d.data_ptr<float>(),
        valid_gaussian.data_ptr<bool>(),
        cov_inv.data_ptr<float>(),
        radii.data_ptr<int>(),
        ntiles.data_ptr<int>(),
        depths.data_ptr<float>(),
        N,
        width,
        height);

    uint32_t *offset;
    int *d_temp_storage;
    cudaMalloc(&offset, N * sizeof(uint32_t));
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveSum(
        nullptr,                // Temporary storage buffer
        temp_storage_bytes,     // Size of temporary storage
        ntiles.data_ptr<int>(), // Input data on the device
        offset,                 // Output result on the device
        N);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceScan::InclusiveSum(
        d_temp_storage,         // Temporary storage buffer
        temp_storage_bytes,     // Size of temporary storage
        ntiles.data_ptr<int>(), // Input data on the device
        offset,                 // Output result on the device
        N);

    uint64_t *keys;
    uint32_t *total_keys = new uint32_t;
    cudaMemcpy(total_keys, offset + N - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMalloc(&keys, (*total_keys) * sizeof(uint64_t));
    uint64_t *indices;

    cudaMalloc(&indices, (*total_keys) * sizeof(uint64_t));


    duplicate_and_assign_key_kernel<<<(N + 1023) / 1024, 1024>>>(
        offset,
        means2d.data_ptr<float>(),
        depths.data_ptr<float>(),
        viewmat.data_ptr<float>(),
        radii.data_ptr<int>(),
        K.data_ptr<float>(),
        N,
        keys,   // Out
        indices // Out
    );

    uint64_t *keys_sorted;
    uint64_t *indices_sorted;
    cudaMalloc(&keys_sorted, (*total_keys) * sizeof(uint64_t));
    cudaMalloc(&indices_sorted, (*total_keys) * sizeof(uint64_t));

    // Sort the keys and indices
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        temp_storage_bytes,
        keys,
        keys_sorted,
        indices,
        indices_sorted,
        *total_keys);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Sort the keys and indices
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        keys,
        keys_sorted,
        indices,
        indices_sorted,
        *total_keys);

    uint32_t *tiles_range_start;
    uint32_t *tiles_range_end;
    int n_tiles = (width + 15) / 16 * (height + 15) / 16;
    cudaMalloc(&tiles_range_start, n_tiles * sizeof(uint32_t));
    cudaMalloc(&tiles_range_end, n_tiles * sizeof(uint32_t));

    identify_tiles_kernel<<<((*total_keys) + 1023) / 1024, 1024>>>(
        keys_sorted,
        indices_sorted,
        *total_keys,
        tiles_range_start,
        tiles_range_end,
        width,
        height);



    torch::Tensor out = torch::zeros({height, width, 3}).cuda();


    out = out.to(torch::kFloat32);

    render_kernel_tiled<<<blocks, threads2>>>(
        means2d.data_ptr<float>(),
        scales.data_ptr<float>(),
        opacities.data_ptr<float>(),
        colors.data_ptr<float>(),
        valid_gaussian.data_ptr<bool>(),
        cov_inv.data_ptr<float>(),
        radii.data_ptr<int>(),
        out.data_ptr<float>(),
        tiles_range_start,
        tiles_range_end,
        indices_sorted,
        width,
        height,
        N);
    return out;
}