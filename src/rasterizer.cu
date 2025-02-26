#include <torch/extension.h>
#include <cub/cub.cuh>

template <int M, int N, int K>
__forceinline__ __device__ void matmul(float *A, float *B, float *C)
{
    #pragma unroll
    for (int i = 0; i < M; i++)
    {
        #pragma unroll
        for (int j = 0; j < K; j++)
        {
            float sum = 0;
            #pragma unroll
            for (int l = 0; l < N; l++)
            {
                sum += A[i * N + l] * B[l * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

using float9 = float[9];

template <int M, int N>
__forceinline__ __device__ void transpose(float *A, float *B)//, int m, int n)
{
    #pragma unroll
    for (int i = 0; i < M; i++)
    {
        #pragma unroll
        for (int j = 0; j < N; j++)
        {
            B[j * M + i] = A[i * N + j];
        }
    }
}

__forceinline__ __device__  void calc_cov3d(const float * __restrict__ scale, float * __restrict__ quat, float * __restrict__ cov3d)
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
    matmul<3,3,3>((float *)R, (float *)S, (float *)RS);
    float RST[3][3];
    transpose<3,3>((float *)RS, (float *)RST);
    // float *cov3d = new float[9];
    // float cov3d[3][3];
    matmul<3,3,3>((float *)RS, (float *)RST, (float *)cov3d);
    // return (float*)cov3d;
}

__device__ float calc_radius(const float cov[2][2])
{
    float det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    float mid = 0.5f * (cov[0][0] + cov[1][1]);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float radius = 1*ceil(3.f * sqrt(max(lambda1, lambda2)));
    return radius;
}

struct Rect
{
    int x0, y0, x1, y1;
};

__forceinline__ __device__ Rect get_rect(const int x, const int y, const int radius, const int width, const int height)
{
    /*
    This function will give the rectangle of the tiles that are touched by the gaussian
    x0, y0 is the top left corner of the rectangle
    x1, y1 is the bottom right corner of the rectangle
    */
    int x0 = min((width + 15) / 16, max(0, (int)((x - radius) / 16)));
    int x1 = min((width + 15) / 16, max(0, (int)((x + radius + 15) / 16)));
    int y0 = min((height + 15) / 16, max(0, (int)((y - radius) / 16)));
    int y1 = min((height + 15) / 16, max(0, (int)((y + radius + 15) / 16)));
    return {x0, y0, x1, y1};
}

__global__ void preprocess_kernel(const float3 * __restrict__ means3d, const float3 * __restrict__ scales, const float4 * __restrict__ quats, const float * __restrict__ viewmat, const float * __restrict__ K, float * __restrict__ means2d, bool * __restrict__ valid_gaussian, float * __restrict__ cov_inv, int *__restrict__ radii, int * __restrict__ ntiles, float * __restrict__ depths, const int N, const int width, const int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
    {
        return;
    }
    float3 means3d_current = means3d[idx];
    float x = means3d_current.x;
    float y = means3d_current.y;
    float z = means3d_current.z;
    float x2 = viewmat[0] * x + viewmat[1] * y + viewmat[2] * z + viewmat[3];
    float y2 = viewmat[4] * x + viewmat[5] * y + viewmat[6] * z + viewmat[7];
    float z2 = viewmat[8] * x + viewmat[9] * y + viewmat[10] * z + viewmat[11];
    float x3 = K[0] * x2 + K[1] * y2 + K[2] * z2;
    float y3 = K[3] * x2 + K[4] * y2 + K[5] * z2;
    means2d[idx * 2] = x3 / z2;
    means2d[idx * 2 + 1] = y3 / z2;
    valid_gaussian[idx] = (z2 > 0.01) && means2d[idx * 2] >= 0 && means2d[idx * 2] < (width) && means2d[idx * 2 + 1] >= 0 && means2d[idx * 2 + 1] < (height + 0);

    // if (!valid_gaussian[idx])
    // {
    //     return;
    // }

    float fx = K[0];
    float fy = K[4];

    float scale[3] = {scales[idx].x, scales[idx].y, scales[idx].z};
    // float quat[4] = {quats[idx * 4], quats[idx * 4 + 1], quats[idx * 4 + 2], quats[idx * 4 + 3]};
    float quat[4] = {quats[idx].x, quats[idx].y, quats[idx].z, quats[idx].w};
    float cov3d[3][3];
    calc_cov3d(scale, quat, (float*)cov3d);

    float J[2][3] = {
        {fx / z2, 0, -fx * x2 / (z2 * z2)},
        {0, fy / z2, -fy * y2 / (z2 * z2)}};

    float W[3][3] = {
        {viewmat[0], viewmat[1], viewmat[2]},
        {viewmat[4], viewmat[5], viewmat[6]},
        {viewmat[8], viewmat[9], viewmat[10]}};

    float WT[3][3];
    transpose<3,3>((float *)W, (float *)WT);
    float JT[3][2];
    transpose<2,3>((float *)J, (float *)JT);

    float WJT[3][2];
    matmul<3,3,2>((float *)WT, (float *)JT, (float *)WJT);
    float JW[2][3];
    matmul<2,3,3>((float *)J, (float *)W, (float *)JW);

    float JWCov[2][3];
    matmul<2,3,3>((float *)JW, (float *)cov3d, (float *)JWCov);

    float cov[2][2];
    matmul<2,3,2>((float *)JWCov, (float *)WJT, (float *)cov);
    // delete cov3D;

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

    if (!valid_gaussian[idx])
    {
        if (ntiles[idx] < 0)
        {
            ntiles[idx] = 0;
        }
        ntiles[idx] = 0;
        depths[idx] = 0;
        radii[idx] = 0;
    }

}



__global__ void __launch_bounds__(256) render_kernel_tiled(const float2  * __restrict__ means2d, const float   * __restrict__ opacities, const float3  * __restrict__ color, const bool * __restrict__ valid_gaussian, const float4  * __restrict__ cov_inv, const int  * __restrict__ radii, float3 * __restrict__ out, const uint32_t  * __restrict__ tiles_range_start, const uint32_t * __restrict__ tiles_range_end, const uint64_t * __restrict__ indices_sorted, const int width, const int height, const int N)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    bool done = false;
    bool inside = true;
    if (x >= width || y >= height)
    {
        done = true;
        inside = false;
    }

    float channel[3] = {0.0, 0.0, 0.0};
    int tile_x = x / 16;
    int tile_y = y / 16;
    int tile_id = tile_y * ((width + 15) / 16) + tile_x;
    int offset = tiles_range_start[tile_id];
    int end = tiles_range_end[tile_id];


    __shared__ int tile_indices[256];
    __shared__ float2 tile_means2d[256];
    __shared__ float4 tile_cov_inv[256];
    __shared__ int tile_radii[256];
    __shared__ float3 tile_color[256];
    __shared__ float tile_opacities[256];
    __shared__ bool tile_valid_gaussian[256];
    float T = 1.0;
    for (int i = offset; i < end; i += 256)
    {
        if (256 == __syncthreads_count(done))
        {
            break;
        }
        int sorted_idx = i + threadIdx.x + threadIdx.y * 16;
        tile_valid_gaussian[threadIdx.x + threadIdx.y * 16] = false;
        if (sorted_idx < end)
        {
            // continue;
        // }
            int idx = indices_sorted[sorted_idx];
            // printf("idx: %d\n", idx);
            const int base_idx = threadIdx.x + threadIdx.y * 16;
            tile_indices[base_idx] = idx;
            tile_means2d[base_idx] = means2d[idx];
            // tile_means2d[(base_idx) * 2 + 1] = means2d[idx * 2 + 1];
            // tile_cov_inv[(base_idx) * 4] = cov_inv[idx * 4];
            // tile_cov_inv[(base_idx) * 4 + 1] = cov_inv[idx * 4 + 1];
            // tile_cov_inv[(base_idx) * 4 + 2] = cov_inv[idx * 4 + 2];
            // tile_cov_inv[(base_idx) * 4 + 3] = cov_inv[idx * 4 + 3];
            tile_cov_inv[base_idx] = cov_inv[idx];
            tile_radii[base_idx] = radii[idx];
            // tile_color[(base_idx) * 3] = color[idx * 3];
            // tile_color[(base_idx) * 3 + 1] = color[idx * 3 + 1];
            // tile_color[(base_idx) * 3 + 2] = color[idx * 3 + 2];
            tile_color[base_idx] = color[idx];
            tile_opacities[base_idx] = opacities[idx];
            tile_valid_gaussian[base_idx] = valid_gaussian[idx];
        }
        __syncthreads();    
        if (done) {
            continue;
        }
        for (int j = 0; j < 256; ++j)
        {
            if (i + j >= end)
            {
                continue;
            }
            if (!tile_valid_gaussian[j])
            {
                continue;
            }
            float x0 = tile_means2d[j].x;
            float y0 = tile_means2d[j].y;
            float delta_x = x - x0;
            float delta_y = y - y0;
            // float aa = tile_cov_inv[j * 4 + 0];
            // float bb = tile_cov_inv[j * 4 + 1];
            // float cc = tile_cov_inv[j * 4 + 2];
            // float dd = tile_cov_inv[j * 4 + 3];
            float aa = tile_cov_inv[j].x;
            float bb = tile_cov_inv[j].y;
            float cc = tile_cov_inv[j].z;
            float dd = tile_cov_inv[j].w;

            float power = -0.5 * (aa * delta_x * delta_x + bb * delta_x * delta_y + cc * delta_y * delta_x + dd * delta_y * delta_y);
            if (power > 0.0)
            {
                continue;
            }

            float alpha = tile_opacities[j] * exp(power);
            alpha = min(0.99f, alpha);
            if (alpha < 1.0f / 255.0f)
				continue;
            // alpha = max(0.00f, alpha);
            channel[0] += T * alpha * tile_color[j].x;
            channel[1] += T * alpha * tile_color[j].y;
            channel[2] += T * alpha * tile_color[j].z;

            T = T * (1 - alpha);
            if (T < 0.0001)
            {
                done = true;
                break;
            }
        }
    }
    if (inside)
    {
        // out[(y * width + x)].x = channel[0];
        // out[(y * width + x)].y = channel[1];
        // out[(y * width + x)].z = channel[2];
        out[y*width+x] = {channel[0], channel[1], channel[2]};
    }
}

__global__ void duplicate_and_assign_key_kernel(
    const uint32_t * __restrict__ offset,
    const float *__restrict__ means2d,
    const float *__restrict__ depths,
    const float *__restrict__ viewmat,
    const int *__restrict__ radii,
    const float *__restrict__ K,
    const int N,
    const int width,
    const int height,
    uint64_t *__restrict__ keys,   // Out
    uint64_t *__restrict__ indices // Out
)
{
    const uint32_t total_indices = offset[N - 1];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
    {
        return;
    }
    if (radii[idx] <= 0)
    {
        return;
    }

    // int width = K[2] * 2;
    // int height = K[5] * 2;
    Rect rect = get_rect(means2d[idx * 2], means2d[idx * 2 + 1], radii[idx], width, height);
    int off = (idx == 0 ? 0 : offset[idx - 1]);
    for (int i = rect.y0; i < rect.y1; ++i)
    {
        for (int j = rect.x0; j < rect.x1; ++j)
        {
            uint64_t key = i * ((width + 15) / 16) + j;
            key = (key << 32);
            key = (key | *((uint32_t *)(&depths[idx])));
            keys[off] = key;
            indices[off] = idx;
            off++;
            if (off >= offset[idx])
            {
                break;
            }

        }
    }
}

__global__ void identify_tiles_kernel(
    const uint64_t * __restrict__ keys_sorted,
    const uint64_t *__restrict__ indices_sorted,
    const uint32_t total_keys,
    uint32_t *__restrict__  tiles_range_start,
    uint32_t *__restrict__ tiles_range_end,
    const int width,
    const int height)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_keys)
    {
        return;
    }
    unsigned int tile = keys_sorted[idx] >> 32;
    if (idx == 0)
    {
        tiles_range_start[0] = 0;
    }
    else
    {
        unsigned int prev_tile = keys_sorted[idx - 1] >> 32;
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    preprocess_kernel<<<(N + 256 - 1) / 256, 256>>>(
        (float3*)means3d.data_ptr<float>(),
        (float3*)scales.data_ptr<float>(),
        (float4*)quats.data_ptr<float>(),
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

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("preprocess kernel execution time: %f ms\n", elapsedTime);

    

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
        width,
        height,
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
    cudaFree(d_temp_storage);
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
    int n_tiles = ((width + 15) / 16) * ((height + 15) / 16);
    cudaMalloc(&tiles_range_start, n_tiles * sizeof(uint32_t));
    cudaMalloc(&tiles_range_end, n_tiles * sizeof(uint32_t));


    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start, 0);

    identify_tiles_kernel<<<((*total_keys) + 1023) / 1024, 1024>>>(
        keys_sorted,
        indices_sorted,
        *total_keys,
        tiles_range_start,
        tiles_range_end,
        width,
        height);


    // cudaEventRecord(stop, 0);

    // cudaEventSynchronize(stop);
    // float elapsedTime;
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf("Identify tiles kernel execution time: %f ms\n", elapsedTime);



    torch::Tensor out = torch::zeros({height, width, 3}).cuda();


    out = out.to(torch::kFloat32);

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2, 0);


    render_kernel_tiled<<<blocks, threads2>>>(
        (float2*)means2d.data_ptr<float>(),
        opacities.data_ptr<float>(),
        (float3*)colors.data_ptr<float>(),
        valid_gaussian.data_ptr<bool>(),
        (float4*)cov_inv.data_ptr<float>(),
        radii.data_ptr<int>(),
        (float3*)out.data_ptr<float>(),
        tiles_range_start,
        tiles_range_end,
        indices_sorted,
        width,
        height,
        N);

    cudaEventRecord(stop2, 0);

    cudaEventSynchronize(stop2);

    cudaEventElapsedTime(&elapsedTime, start2, stop2);

    printf("Render kernel execution time: %f ms\n", elapsedTime);

    

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    cudaFree(offset);
    cudaFree(d_temp_storage);
    cudaFree(keys);
    cudaFree(indices);
    cudaFree(keys_sorted);
    cudaFree(indices_sorted);
    cudaFree(tiles_range_start);
    cudaFree(tiles_range_end);

    return out;
}