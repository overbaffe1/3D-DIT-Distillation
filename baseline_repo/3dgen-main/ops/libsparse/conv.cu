
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

#define KERNEL_SIZE 3
#define DELTA_SIZE (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE)

__device__ __forceinline__ uint64_t split_by_3_64(int x) {
    uint64_t w = static_cast<uint64_t>(x);
    w &=                0x00000000001fffffULL;
    w = (w | w << 32) & 0x001f00000000ffffULL;
    w = (w | w << 16) & 0x001f0000ff0000ffULL;
    w = (w | w <<  8) & 0x010f00f00f00f00fULL;
    w = (w | w <<  4) & 0x10c30c30c30c30c3ULL;
    w = (w | w <<  2) & 0x1249249249249249ULL;
    return w;
}

__device__ __forceinline__ int64_t get_order_code(const int4 coord){
    uint64_t key = static_cast<uint64_t>(coord.x) << 36;
//     key |= (split_by_3_64(coord.y) << 2) | (split_by_3_64(coord.z) << 1) | (split_by_3_64(coord.w) << 0);
    key |= (static_cast<uint64_t>(coord.y) << 24) | (static_cast<uint64_t>(coord.z) << 12) | (static_cast<uint64_t>(coord.w) << 0);
    return static_cast<int64_t>(key);
}

__device__ __forceinline__ int32_t compact_by_3_64(uint64_t w) {
    w &=                  0x1249249249249249ULL;
    w = (w ^ (w >> 2))  & 0x30c30c30c30c30c3ULL;
    w = (w ^ (w >> 4))  & 0xf00f00f00f00f00fULL;
    w = (w ^ (w >> 8))  & 0x00ff0000ff0000ffULL;
    w = (w ^ (w >> 16)) & 0x00ff00000000ffffULL;
    w = (w ^ (w >> 32)) & 0x00000000001fffffULL;
    return static_cast<int32_t>(w & 0xFFFULL);
}

__device__ __forceinline__ int4 inverse_order_code(int64_t key){
    int32_t bidx = static_cast<uint64_t>(key) >> 36;

//     int32_t x = compact_by_3_64(key >> 2);
//     int32_t y = compact_by_3_64(key >> 1);
//     int32_t z = compact_by_3_64(key >> 0);

    int32_t x = (static_cast<uint64_t>(key) >> 24) & 0xFFFULL;
    int32_t y = (static_cast<uint64_t>(key) >> 12) & 0xFFFULL;
    int32_t z = (static_cast<uint64_t>(key) >> 0) & 0xFFFULL;

    return make_int4(bidx, x, y, z);
}

__global__ void pos2key_kernel(
    const int N,
    const int4* __restrict__ pos,
    int64_t* __restrict__ keys
){
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(gidx >= N) return;

    const int4 coord = pos[gidx];
    keys[gidx] = get_order_code(coord);
}

__device__ inline int3 offset_of(int d) {

  int rem = d;
  int dz = rem % KERNEL_SIZE - (KERNEL_SIZE - 1) / 2; rem /= KERNEL_SIZE;
  int dy = rem % KERNEL_SIZE - (KERNEL_SIZE - 1) / 2; rem /= KERNEL_SIZE;
  int dx = rem % KERNEL_SIZE - (KERNEL_SIZE - 1) / 2;

  return make_int3(dx, dy, dz);
}

__global__ void delta_indices_kernel(
    const int64_t N, const int64_t N_ALIGNED,
    const int64_t* __restrict__ query_keys,
    const int64_t* __restrict__ sorted_keys,
    const int64_t* __restrict__ order,
    int* __restrict__ fwd_indices
){
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(gidx >= N) return;

    const int4 coord = inverse_order_code(query_keys[gidx]);
    int bidx = coord.x, x = coord.y, y = coord.z, z = coord.w;

    // precompute bound for further searches
    int3 off_min = offset_of(0), off_max = offset_of(DELTA_SIZE - 1);
    int64_t key_min = get_order_code(make_int4(bidx, max(x + off_min.x, 0), max(y + off_min.y, 0), max(z + off_min.z, 0)));
    int64_t key_max = get_order_code(make_int4(bidx, min(x + off_max.x, (1 << 12) - 1), min(y + off_max.y, (1 << 12) - 1), min(z + off_max.z, (1 << 12) - 1)));

    int low0 = 0, low1 = N;
    while(low1 - low0 > 1){
        int mid = (low0 + low1) >> 1;
        if(__ldg(&sorted_keys[mid]) <= key_min) low0 = mid;
        else low1 = mid;
    }

    int hgh0 = 0, hgh1 = N;
    while(hgh1 - hgh0 > 1){
        int mid = (hgh0 + hgh1) >> 1;
        if(__ldg(&sorted_keys[mid]) <= key_max) hgh0 = mid;
        else hgh1 = mid;
    }

    #pragma unroll
    for(int d = 0; d < DELTA_SIZE; d++){


        const int3 off = offset_of(d);
        const int xx = x + off.x, yy = y + off.y, zz = z + off.z;

        int found_idx = -1;
        if(xx >= 0 && yy >= 0 && zz >= 0) {

            int64_t adjkey = get_order_code(make_int4(bidx, xx, yy, zz));

            int l = low0, r = hgh1;
            while(r - l > 1){
                int mid = (r + l) >> 1;
                if(__ldg(&sorted_keys[mid]) <= adjkey) l = mid;
                else r = mid;
            }

            if (sorted_keys[l] == adjkey) found_idx = static_cast<int>(order[l]);

        }

        fwd_indices[gidx] = found_idx;
        fwd_indices += N_ALIGNED;
    }

}

std::vector<torch::Tensor> conv_indices_cuda(torch::Tensor pos){

    const int threads = 512;
    auto device = pos.device();

    const int64_t N = pos.size(0);
    const int64_t N_ALIGNED = (N + 3) / 4 * 4;

    auto keys = torch::empty({N}, torch::TensorOptions().device(device).dtype(torch::kInt64));
    pos2key_kernel<<<(N + threads - 1) / threads, threads>>>(
        N,
        reinterpret_cast<const int4*>(pos.data_ptr<int>()),
        keys.data_ptr<int64_t>()
    );

    torch::Tensor order = torch::argsort(keys);
    torch::Tensor sorted_keys = keys.index({order});

    auto fwd_indices = torch::empty({DELTA_SIZE, N_ALIGNED}, torch::TensorOptions().device(device).dtype(torch::kInt32));
    delta_indices_kernel<<<(N + threads - 1) / threads, threads>>>(
        N, N_ALIGNED,
        keys.data_ptr<int64_t>(),
        sorted_keys.data_ptr<int64_t>(),
        order.data_ptr<int64_t>(),
        fwd_indices.data_ptr<int>()
    );

    return {fwd_indices.t().contiguous()};
}

int get_number_sm(torch::Device device) {
    int nsm = 0;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, device.index());
    return nsm;
}

template <typename T, int N>
struct alignas(16) AlignedVector { T val[N]; };

template <typename scalar_t>
struct VecTraits {
    static constexpr int kElems = 16 / sizeof(scalar_t);
    using Vec = AlignedVector<scalar_t, kElems>;
    __device__ __forceinline__ static Vec load(const scalar_t* __restrict__ p) {
        return *reinterpret_cast<const Vec*>(p);
    }
    __device__ __forceinline__ static void store(scalar_t* __restrict__ p, const Vec& v) {
        *reinterpret_cast<Vec*>(p) = v;
    }
};


template <typename scalar_t>
__global__ void conv_fwd_kernel(
    const int64_t N, const int C, const int64_t N_CHUNKS, const int C_CHUNKS,
    const int N_TILE, const int C_TILE, const int64_t WORK_PER_GROUP,
    const scalar_t* __restrict__ feat, const int* __restrict__ fwd_indices,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    scalar_t* __restrict__ out
) {

    using VT = VecTraits<scalar_t>;
    constexpr int VEC_SIZE = VT::kElems;

    extern __shared__ uint8_t cache[];
    float* weight_cache = reinterpret_cast<float*>(cache);
    int* item_indices = reinterpret_cast<int*>(cache + (DELTA_SIZE + 1) * C_TILE * VEC_SIZE * sizeof(float));

    const int THREADS = blockDim.x;
    const size_t GROUP_IDX = blockIdx.x;
    const int LOCAL_IDX = threadIdx.x;

    const size_t C_GROUP = GROUP_IDX % C_CHUNKS;
    const size_t N_GROUP = GROUP_IDX / C_CHUNKS;
    const int C_IDX = LOCAL_IDX % C_TILE;
    const int N_IDX = LOCAL_IDX / C_TILE;

    const int CHANNEL_START = C_GROUP * C_TILE * VEC_SIZE;
    const int CHANNEL_BASE = CHANNEL_START + C_IDX * VEC_SIZE;

    // load weight into local cache
    for (int d = 0; d < DELTA_SIZE; d++){
        for(int k = LOCAL_IDX; k < C_TILE * VEC_SIZE; k += THREADS){
            int channel_idx = CHANNEL_START + k;
            if(channel_idx < C) weight_cache[d * C_TILE * VEC_SIZE + (k % VEC_SIZE) * C_TILE + k / VEC_SIZE] = weight[static_cast<size_t>(d) * C + channel_idx];
        }
    }
    for(int k = LOCAL_IDX; k < C_TILE * VEC_SIZE; k += THREADS){
        int channel_idx = CHANNEL_START + k;
        if(channel_idx < C) weight_cache[DELTA_SIZE * C_TILE * VEC_SIZE + (k % VEC_SIZE) * C_TILE + k / VEC_SIZE] = bias[channel_idx];
    }
//     __syncthreads();

    float acc[VEC_SIZE];
    for(size_t TILE_IDX = N_GROUP; TILE_IDX < N_CHUNKS; TILE_IDX += (N_CHUNKS + WORK_PER_GROUP - 1) / WORK_PER_GROUP){

        for (int d = LOCAL_IDX; d < DELTA_SIZE * N_TILE; d += THREADS) {
            size_t item_idx = (TILE_IDX * N_TILE + d / DELTA_SIZE);
            if(item_idx < N) item_indices[(d % DELTA_SIZE) * N_TILE + d / DELTA_SIZE] = fwd_indices[static_cast<size_t>(item_idx) * DELTA_SIZE + (d % DELTA_SIZE)];
        }
        __syncthreads();

        const size_t item_idx = TILE_IDX * N_TILE + N_IDX;
        if (item_idx < N && CHANNEL_BASE < C){
            #pragma unroll
            for(int k = 0; k < VEC_SIZE; k++) {
                acc[k] = weight_cache[DELTA_SIZE * C_TILE * VEC_SIZE + k * C_TILE + C_IDX];
            }

            for(int d = 0; d < DELTA_SIZE; d++){
                int prev_item = item_indices[d * N_TILE + N_IDX];
                if(prev_item < 0) continue;

                AlignedVector<scalar_t, VEC_SIZE> vfeat = VT::load(feat + (static_cast<size_t>(prev_item) * C + CHANNEL_BASE));
                #pragma unroll
                for(int k = 0; k < VEC_SIZE; k++){
                    acc[k] = fmaf(static_cast<float>(vfeat.val[k]), weight_cache[d * C_TILE * VEC_SIZE + k * C_TILE + C_IDX], acc[k]);
                }
            }

            AlignedVector<scalar_t, VEC_SIZE> ofeat;
            #pragma unroll
            for(int k = 0; k < VEC_SIZE; k++) ofeat.val[k] = static_cast<scalar_t>(acc[k]);
            VT::store(out + (static_cast<size_t>(item_idx) * C + CHANNEL_BASE), ofeat);
        }
        __syncthreads();
    }

}

torch::Tensor conv_fwd_cuda(torch::Tensor feat, torch::Tensor weight, torch::Tensor bias, std::vector<torch::Tensor> indices){

    auto device = feat.device();
    const int SM_COUNT = get_number_sm(device);

    const int64_t N = feat.size(0);
    const int64_t N_ALIGNED = (N + 3) / 4 * 4;
    const int C = feat.size(1);

    TORCH_CHECK(weight.size(0) == KERNEL_SIZE && weight.size(1) == KERNEL_SIZE && weight.size(2) == KERNEL_SIZE, "wrong weight shape");

    auto fwd_indices = indices[0];
    TORCH_CHECK(fwd_indices.dim() == 2 && fwd_indices.size(0) == N_ALIGNED && fwd_indices.size(1) == DELTA_SIZE && fwd_indices.dtype() == torch::kInt32, "wrong idx shape");

    // single load size
    const int VEC_SIZE = 16 / feat.element_size();
    const int C_TILE = std::min(256 / VEC_SIZE, C / VEC_SIZE); // <= 64
    int N_TILE = 1;
    while((C_TILE * N_TILE) % 32 > 0 || (C_TILE * N_TILE) <= 128) N_TILE *= 2; // make it divisible to 32 and at least 256
    const int THREADS = C_TILE * N_TILE;

    // one group in one iteration process C_TILE * VEC_SIZE * N_TILE items
    const int C_CHUNKS = (C + C_TILE * VEC_SIZE - 1) / (C_TILE * VEC_SIZE);
    const int64_t N_CHUNKS = (N + N_TILE - 1) / N_TILE;

    // how much tiles to process per group such that all SMs are occupied
    const int64_t WORK_PER_GROUP = std::max((N_CHUNKS * C_CHUNKS + 4 * SM_COUNT - 1) / (4 * SM_COUNT), static_cast<int64_t>(1));
    const int64_t GROUPS = (N_CHUNKS + WORK_PER_GROUP - 1) / WORK_PER_GROUP * C_CHUNKS;

    auto out = torch::empty_like(feat);

    const int SHARED_MEM = (DELTA_SIZE + 1) * C_TILE * VEC_SIZE * sizeof(float) + DELTA_SIZE * N_TILE * sizeof(int);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, feat.scalar_type(), "conv_fwd_kernel", [&]{
        conv_fwd_kernel<scalar_t><<<GROUPS, THREADS, SHARED_MEM>>>(
            N, C, N_CHUNKS, C_CHUNKS, N_TILE, C_TILE, WORK_PER_GROUP,
            feat.data_ptr<scalar_t>(), fwd_indices.data_ptr<int>(),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            out.data_ptr<scalar_t>()
        );
    });


    return out;
}

#define DW_THREADS 256

template <typename scalar_t>
__global__ void conv_dw_kernel(
    const int64_t N, const int C, const int64_t N_CHUNKS, const int C_CHUNKS, const int64_t WORK_PER_GROUP, const int64_t N_ALIGNED,
    const scalar_t* __restrict__ feat, const int* __restrict__ fwd_indices,
    const scalar_t* __restrict__ grad_out,
    float* __restrict__ grad_dw
) {

    using VT = VecTraits<scalar_t>;
    constexpr int VEC_SIZE = VT::kElems;
    constexpr int N_WARPS = (DW_THREADS + 31) >> 5;

    const unsigned warp_mask = __activemask();
    const int warp_lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    const int THREADS = blockDim.x;
    const size_t GROUP_IDX = blockIdx.x;
    const size_t LOCAL_IDX = threadIdx.x;

    const size_t C_IDX = GROUP_IDX % C_CHUNKS;
    const size_t N_IDX = GROUP_IDX / C_CHUNKS;

    float acc[VEC_SIZE * DELTA_SIZE];
    #pragma unroll
    for(int k = 0; k < DELTA_SIZE * VEC_SIZE; k++) acc[k] = 0.0f;

    //     __shared__ int item_indices[DELTA_SIZE * DW_THREADS];
    for(size_t TILE_IDX = N_IDX; TILE_IDX < N_CHUNKS; TILE_IDX += (N_CHUNKS + WORK_PER_GROUP - 1) / WORK_PER_GROUP){

//         for(int idx = LOCAL_IDX; idx < DELTA_SIZE * THREADS / 4; idx += THREADS){
//
//             const int base = idx % (THREADS / 4), d = idx / (THREADS / 4);
//             const int item_base = TILE_IDX * THREADS + base * 4;
//
//             AlignedVector<int, 4> indices_vec;
//             if(item_base < N_ALIGNED) indices_vec = VecTraits<int>::load(fwd_indices + d * N_ALIGNED + item_base);
//
//             #pragma unroll
//             for(int k = 0; k < 4; k++) item_indices[d * THREADS + base * 4 + k] = indices_vec.val[k];
//         }
//         __syncthreads();

        const size_t item_idx = TILE_IDX * THREADS + LOCAL_IDX;
        if (item_idx < N){

            AlignedVector<scalar_t, VEC_SIZE> grad_out_vec = VT::load(grad_out + (static_cast<size_t>(C_IDX) * N * VEC_SIZE + static_cast<size_t>(item_idx) * VEC_SIZE));

            for(int d = 0; d < DELTA_SIZE; d++){

//                 int prev_item = item_indices[d * THREADS + LOCAL_IDX];
                int prev_item = fwd_indices[static_cast<size_t>(d) * N_ALIGNED + item_idx];
                if(prev_item < 0) continue;

                AlignedVector<scalar_t, VEC_SIZE> feat_vec = VT::load(feat + (static_cast<size_t>(C_IDX) * N * VEC_SIZE + static_cast<size_t>(prev_item) * VEC_SIZE));
                #pragma unroll
                for(int k = 0; k < VEC_SIZE; k++){
                    acc[d * VEC_SIZE + k] = fmaf(static_cast<float>(feat_vec.val[k]), static_cast<float>(grad_out_vec.val[k]), acc[d * VEC_SIZE + k]);
                }

            }
        }

        __syncthreads(); // for item_indices sync
    }



//     firstly reduce sum inside each warp
    __shared__ float shared[N_WARPS * VEC_SIZE * DELTA_SIZE];
    for(int r = 0; r < VEC_SIZE * DELTA_SIZE; r++){
        float v = acc[r];
        for(int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(warp_mask, v, off);
        if (warp_lane == 0) shared[warp_id * VEC_SIZE * DELTA_SIZE + r] = v;
    }
    __syncthreads();

//     reduce sums from all warps by each thread
    __shared__ float shared_acc[VEC_SIZE * DELTA_SIZE];
    for(int r = LOCAL_IDX; r < VEC_SIZE * DELTA_SIZE; r += THREADS){
        float v = 0.0f;
        #pragma unroll
        for(int w = 0; w < N_WARPS; w++) v += shared[w * VEC_SIZE * DELTA_SIZE + r];
        shared_acc[r] = v;
    }
    __syncthreads();

    if constexpr (VEC_SIZE < 4){
        for(int idx = LOCAL_IDX; idx < DELTA_SIZE * VEC_SIZE; idx += THREADS){
            int d = idx / VEC_SIZE, vec_idx = idx % VEC_SIZE;
            grad_dw[static_cast<size_t>(N_IDX) * C_CHUNKS * DELTA_SIZE * VEC_SIZE + static_cast<size_t>(C_IDX) * DELTA_SIZE * VEC_SIZE + d * VEC_SIZE + vec_idx] = shared_acc[d * VEC_SIZE + vec_idx];
        }
    } else {
        AlignedVector<float, 4> out_vec;
        for(int idx = LOCAL_IDX; idx < DELTA_SIZE * VEC_SIZE / 4; idx += THREADS){
            int d = idx / (VEC_SIZE / 4), vec_base = (idx % (VEC_SIZE / 4)) * 4;
            #pragma unroll
            for(int k = 0; k < 4; k++){
                out_vec.val[k] = shared_acc[d * VEC_SIZE + vec_base + k];
            }
            VecTraits<float>::store(grad_dw + (static_cast<size_t>(N_IDX) * C_CHUNKS * DELTA_SIZE * VEC_SIZE + static_cast<size_t>(C_IDX) * DELTA_SIZE * VEC_SIZE + d * VEC_SIZE + vec_base), out_vec);
        }
    }

}

std::vector<torch::Tensor> conv_dw_cuda(torch::Tensor feat, torch::Tensor grad_out, std::vector<torch::Tensor> indices) {

    auto device = feat.device();
    const int SM_COUNT = get_number_sm(device);

    const int64_t N = feat.size(0);
    const int64_t N_ALIGNED = (N + 3) / 4 * 4;
    const int C = feat.size(1);

    auto fwd_indices = indices[0];
    TORCH_CHECK(fwd_indices.dim() == 2 && fwd_indices.size(0) == N_ALIGNED && fwd_indices.size(1) == DELTA_SIZE && fwd_indices.dtype() == torch::kInt32, "wrong idx shape");

    auto grad_db = torch::sum(grad_out, /*dim=*/0, /*keepdim=*/false, /*dtype=*/torch::kFloat32); // (,C)

    // single load size
    const int VEC_SIZE = 16 / feat.element_size();
    const int THREADS = DW_THREADS;

    feat = feat.view({N, C / VEC_SIZE, VEC_SIZE}).permute({1, 0, 2}).contiguous();
    grad_out = grad_out.view({N, C / VEC_SIZE, VEC_SIZE}).permute({1, 0, 2}).contiguous();

    const int C_CHUNKS = feat.size(0);
    const int64_t N_CHUNKS = (N + THREADS - 1) / THREADS;

    // how much tiles to process per group such that all SMs are occupied
    const int64_t WORK_PER_GROUP = std::max((N_CHUNKS * C_CHUNKS + 4 * SM_COUNT - 1) / (4 * SM_COUNT), static_cast<int64_t>(1));
    const int64_t GROUPS = (N_CHUNKS + WORK_PER_GROUP - 1) / WORK_PER_GROUP * C_CHUNKS;

    fwd_indices = fwd_indices.transpose(0, 1).contiguous();
    auto grad_dw = torch::zeros({(N_CHUNKS + WORK_PER_GROUP - 1) / WORK_PER_GROUP, C / VEC_SIZE, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, VEC_SIZE}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, feat.scalar_type(), "conv_dw_kernel", [&]{
        conv_dw_kernel<scalar_t><<<GROUPS, THREADS>>>(
            N, C, N_CHUNKS, C_CHUNKS, WORK_PER_GROUP, N_ALIGNED,
            feat.data_ptr<scalar_t>(), fwd_indices.data_ptr<int>(),
            grad_out.data_ptr<scalar_t>(),
            grad_dw.data_ptr<float>()
        );
    });

    grad_dw = torch::sum(grad_dw, /*dim=*/0, /*keepdim=*/false, /*dtype=*/torch::kFloat32).movedim(0, -2).reshape({KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, C}).contiguous();

    return {grad_dw, grad_db};
}

// -------------------------- Dense (non-grouped) convolution --------------------------

inline __host__ __device__ int div_up_int(int x, int y) { return (x + y - 1) / y; }

template <typename scalar_t>
__global__ void dense_conv_fwd_naive_kernel(
    const int N, const int C_IN, const int C_OUT,
    const scalar_t* __restrict__ feat,
    const float* __restrict__ weight, // [DELTA, C_OUT, C_IN]
    const float* __restrict__ bias,   // [C_OUT]
    const int* __restrict__ fwd_indices,
    scalar_t* __restrict__ out
) {
    const int n = blockIdx.x;
    const int co = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= N || co >= C_OUT) return;

    float acc = bias[co];
    const int* idx_ptr = fwd_indices + n * DELTA_SIZE;
    #pragma unroll
    for (int d = 0; d < DELTA_SIZE; ++d) {
        const int prev = idx_ptr[d];
        if (prev < 0) continue;
        const scalar_t* feat_ptr = feat + prev * C_IN;
        const float* w_ptr = weight + d * C_OUT * C_IN + co * C_IN;
        for (int ci = 0; ci < C_IN; ++ci) {
            acc = fmaf(static_cast<float>(feat_ptr[ci]), w_ptr[ci], acc);
        }
    }
    out[n * C_OUT + co] = static_cast<scalar_t>(acc);
}

using namespace nvcuda;

// WMMA-accelerated forward for half inputs. Assumes K_TILE=16.
__global__ void dense_conv_fwd_wmma_kernel(
    const int N, const int C_IN, const int C_OUT,
    const half* __restrict__ feat,
    const float* __restrict__ weight, // [DELTA, C_OUT, C_IN]
    const float* __restrict__ bias,   // [C_OUT]
    const int* __restrict__ fwd_indices,
    half* __restrict__ out
) {
    constexpr int M_TILE = 16; // output channels per tile
    constexpr int N_TILE = 16; // items per tile
    constexpr int K_TILE = 16; // input channels per chunk

    const int tile_n = blockIdx.x;
    const int tile_co = blockIdx.y;

    const int n_start = tile_n * N_TILE;
    const int co_start = tile_co * M_TILE;

    wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    __shared__ half a_shared[M_TILE * K_TILE];
    __shared__ half b_shared[K_TILE * N_TILE];
    __shared__ float c_shared[M_TILE * N_TILE];

    // iterate over input channel chunks
    for (int k_base = 0; k_base < C_IN; k_base += K_TILE) {
        const int k_valid = min(K_TILE, C_IN - k_base);

        // iterate over kernel offsets
        #pragma unroll
        for (int d = 0; d < DELTA_SIZE; ++d) {
            // zero tiles to avoid stale data
            for (int tid = threadIdx.x; tid < M_TILE * K_TILE; tid += blockDim.x) {
                a_shared[tid] = __float2half_rn(0.0f);
            }
            for (int tid = threadIdx.x; tid < K_TILE * N_TILE; tid += blockDim.x) {
                b_shared[tid] = __float2half_rn(0.0f);
            }
            __syncthreads();

            // pack A (weights) into shared memory
            for (int tid = threadIdx.x; tid < M_TILE * K_TILE; tid += blockDim.x) {
                const int row = tid / K_TILE;
                const int col = tid - row * K_TILE;
                const int co = co_start + row;
                const int ci = k_base + col;
                if (co < C_OUT && col < k_valid) {
                    const float* w_ptr = weight + d * C_OUT * C_IN + co * C_IN + ci;
                    a_shared[row * K_TILE + col] = __float2half_rn(*w_ptr);
                }
            }
            __syncthreads();

            // pack B (features) into shared memory (col-major layout for WMMA)
            for (int tid = threadIdx.x; tid < K_TILE * N_TILE; tid += blockDim.x) {
                const int row = tid / N_TILE;
                const int col = tid - row * N_TILE;
                const int n = n_start + col;
                const int ci = k_base + row;
                if (n < N && row < k_valid) {
                    const int prev = fwd_indices[n * DELTA_SIZE + d];
                    if (prev >= 0) {
                        const half* feat_ptr = feat + prev * C_IN + ci;
                        // store transposed: col-major KxN
                        b_shared[col * K_TILE + row] = *feat_ptr;
                    }
                }
            }
            __syncthreads();

            if (threadIdx.x < 32) {
                wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, half, wmma::col_major> b_frag;

                wmma::load_matrix_sync(a_frag, a_shared, K_TILE);
                wmma::load_matrix_sync(b_frag, b_shared, K_TILE);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            __syncthreads();
        }
    }

    // write results
    // store accumulator to shared then to output
    if (threadIdx.x < 32) {
        wmma::store_matrix_sync(c_shared, acc_frag, N_TILE, wmma::mem_row_major);
    }
    __syncthreads();

    for (int tid = threadIdx.x; tid < M_TILE * N_TILE; tid += blockDim.x) {
        const int row = tid / N_TILE;
        const int col = tid - row * N_TILE;
        const int co = co_start + row;
        const int n = n_start + col;
        if (co < C_OUT && n < N) {
            float v = c_shared[tid] + bias[co];
            out[n * C_OUT + co] = __float2half_rn(v);
        }
    }
}

torch::Tensor dense_conv_fwd_cuda(
    torch::Tensor feat,    // (N, C_IN)
    torch::Tensor weight,  // (K, K, K, C_IN, C_OUT)
    torch::Tensor bias,    // (C_OUT)
    std::vector<torch::Tensor> indices
) {
    const auto device = feat.device();
    const int N = feat.size(0);

    TORCH_CHECK(weight.size(0) == KERNEL_SIZE && weight.size(1) == KERNEL_SIZE && weight.size(2) == KERNEL_SIZE, "weight must be 3x3x3");
    weight = weight.permute({0, 1, 2, 4, 3}).contiguous(); // to [K,K,K,C_OUT,C_IN]
    const int C_OUT = weight.size(3);
    const int C_IN = weight.size(4);
    TORCH_CHECK(feat.size(1) == C_IN, "feat channel dim must match weight C_IN");

    auto fwd_indices = indices[0];
    TORCH_CHECK(fwd_indices.dim() == 2 && fwd_indices.size(0) == div_up_int(N, 4) * 4 && fwd_indices.size(1) == DELTA_SIZE && fwd_indices.dtype() == torch::kInt32, "wrong idx shape");

    auto out = torch::zeros({N, C_OUT}, torch::TensorOptions().device(device).dtype(feat.dtype()));

    if (feat.scalar_type() == torch::kHalf && weight.scalar_type() == torch::kFloat && bias.scalar_type() == torch::kFloat && (C_IN % 16) == 0 && (C_OUT % 16) == 0) {
        const dim3 grid(div_up_int(N, 16), div_up_int(C_OUT, 16));
        dense_conv_fwd_wmma_kernel<<<grid, 256>>>(
            N, C_IN, C_OUT,
            reinterpret_cast<half*>(feat.data_ptr<at::Half>()),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            fwd_indices.data_ptr<int>(),
            reinterpret_cast<half*>(out.data_ptr<at::Half>())
        );
    } else {
        dim3 block_naive(256);
        dim3 grid_naive(N, div_up_int(C_OUT, block_naive.x));
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, feat.scalar_type(), "dense_conv_fwd_naive", [&]{
            dense_conv_fwd_naive_kernel<scalar_t><<<grid_naive, block_naive>>>(
                N, C_IN, C_OUT,
                feat.data_ptr<scalar_t>(),
                weight.data_ptr<float>(), bias.data_ptr<float>(),
                fwd_indices.data_ptr<int>(),
                out.data_ptr<scalar_t>()
            );
        });
    }

    return out;
}


template <typename scalar_t>
__global__ void dense_conv_dw_kernel_simple(
    const int N, const int C_IN, const int C_OUT,
    const scalar_t* __restrict__ feat,      // (N, C_IN)
    const scalar_t* __restrict__ grad_out,  // (N, C_OUT)
    const int* __restrict__ fwd_indices,
    float* __restrict__ grad_w              // (DELTA, C_OUT, C_IN)
) {
    const int d = blockIdx.x;
    const int co = blockIdx.y;
    const int ci = blockIdx.z * blockDim.x + threadIdx.x;
    if (co >= C_OUT || ci >= C_IN) return;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        const int prev = fwd_indices[n * DELTA_SIZE + d];
        if (prev < 0) continue;
        const float go = static_cast<float>(grad_out[n * C_OUT + co]);
        const float v = static_cast<float>(feat[prev * C_IN + ci]);
        acc = fmaf(go, v, acc);
    }
    atomicAdd(grad_w + d * C_OUT * C_IN + co * C_IN + ci, acc);
}

std::vector<torch::Tensor> dense_conv_dw_cuda(
    torch::Tensor feat,     // (N, C_IN)
    torch::Tensor grad_out, // (N, C_OUT)
    std::vector<torch::Tensor> indices
) {
    const auto device = feat.device();
    const int N = feat.size(0);
    const int C_IN = feat.size(1);
    const int C_OUT = grad_out.size(1);

    auto fwd_indices = indices[0];
    TORCH_CHECK(fwd_indices.dim() == 2 && fwd_indices.size(0) == div_up_int(N, 4) * 4 && fwd_indices.size(1) == DELTA_SIZE && fwd_indices.dtype() == torch::kInt32, "wrong idx shape");

    auto grad_w = torch::zeros({DELTA_SIZE, C_OUT, C_IN}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    auto grad_b = torch::sum(grad_out, /*dim=*/0, /*keepdim=*/false, /*dtype=*/torch::kFloat32);

    dim3 block(128);
    dim3 grid(DELTA_SIZE, C_OUT, div_up_int(C_IN, block.x));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, feat.scalar_type(), "dense_conv_dw_kernel_simple", [&]{
        dense_conv_dw_kernel_simple<scalar_t><<<grid, block>>>(
            N, C_IN, C_OUT,
            feat.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(),
            fwd_indices.data_ptr<int>(),
            grad_w.data_ptr<float>()
        );
    });

    grad_w = grad_w.permute({0, 2, 1}).contiguous(); // (d, C_IN, C_OUT)
    grad_w = grad_w.view({KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, C_IN, C_OUT}).contiguous();
    return {grad_w, grad_b};
}
