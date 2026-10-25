
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


__device__ __forceinline__ int64_t get_order_code(const int4 coord) {
    uint64_t key = static_cast<uint64_t>(coord.x) << 36;
    key |= (static_cast<uint64_t>(coord.y) << 24)
         | (static_cast<uint64_t>(coord.z) << 12)
         | (static_cast<uint64_t>(coord.w) <<  0);
    return static_cast<int64_t>(key);
}

__device__ __forceinline__ int4 inverse_order_code(int64_t key) {
    const uint64_t ukey = static_cast<uint64_t>(key);
    return make_int4(
        static_cast<int32_t>(ukey >> 36),
        static_cast<int32_t>((ukey >> 24) & 0xFFFULL),
        static_cast<int32_t>((ukey >> 12) & 0xFFFULL),
        static_cast<int32_t>((ukey >>  0) & 0xFFFULL)
    );
}


// Path-halving find via atomicMax (from cutil_math.h)
__device__ inline int dsu_find(int u, int32_t* __restrict__ parent) {
    int32_t up = parent[u];
    int32_t v = u, p = up;
    while (v != p) { v = p; p = parent[v]; }
    if (v != up)
        atomicMax(parent + u, v);
    return v;
}

// Basic union — CAS loop, no size tracking (used by CC)
__device__ inline void dsu_union_basic(int a, int b, int32_t* __restrict__ parent) {
    while (true) {
        a = dsu_find(a, parent);
        b = dsu_find(b, parent);
        if (a == b) return;
        int left = a, right = b;
        if (left < right) { left = b; right = a; }
        if (atomicCAS(parent + right, right, left) == right) break;
    }
}

// Lock-based capped union — locks both roots, verifies, updates parent+size atomically
// lock[i]: 0=free, 1=held. Acquired in index order to prevent deadlock.
__device__ inline void dsu_union_capped(
    int a, int b,
    int32_t* __restrict__ parent,
    int32_t* __restrict__ sz,
    int32_t* __restrict__ lock,
    int cap
) {
    while (true) {
        a = dsu_find(a, parent);
        b = dsu_find(b, parent);
        if (a == b) return;

        // Lock in index order to prevent deadlock
        int lo = min(a, b), hi = max(a, b);

        if (atomicCAS(lock + lo, 0, 1) != 0) continue;
        if (atomicCAS(lock + hi, 0, 1) != 0) {
            atomicExch(lock + lo, 0);
            continue;
        }

        // Verify both are still roots (volatile to bypass L1 cache)
        if (*(volatile int32_t*)(parent + lo) != lo ||
            *(volatile int32_t*)(parent + hi) != hi) {
            atomicExch(lock + hi, 0);
            atomicExch(lock + lo, 0);
            continue;
        }

        // Size cap check (volatile reads to see latest values)
        int slo = *(volatile int32_t*)(sz + lo);
        int shi = *(volatile int32_t*)(sz + hi);
        if (slo + shi > cap) {
            atomicExch(lock + hi, 0);
            atomicExch(lock + lo, 0);
            return;
        }

        // Merge: higher-index becomes root (volatile writes to flush to L2)
        atomicExch(parent + lo, hi);
        atomicExch(sz + hi, slo + shi);

        __threadfence();

        atomicExch(lock + hi, 0);
        atomicExch(lock + lo, 0);
        return;
    }
}

// =====================================================================
// 6-neighbor direction tables in constant memory
// =====================================================================

__constant__ int DIRS6_X[6] = { 1, -1,  0,  0,  0,  0 };
__constant__ int DIRS6_Y[6] = { 0,  0,  1, -1,  0,  0 };
__constant__ int DIRS6_Z[6] = { 0,  0,  0,  0,  1, -1 };

// =====================================================================
// Shared helper kernels
// =====================================================================

__global__ void encode_keys_kernel(
    const int N,
    const int4* __restrict__ pos,
    int64_t* __restrict__ keys
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    keys[i] = get_order_code(pos[i]);
}

// Invert permutation: pos_in_sorted[order[p]] = p
__global__ void invert_perm_kernel(
    const int N,
    const int64_t* __restrict__ order,
    int32_t* __restrict__ pos_in_sorted
) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;
    pos_in_sorted[static_cast<int>(order[p])] = p;
}

// Build 6-neighbor table via binary search in sorted_keys, restricted to batch range
__global__ void build_6nbr_kernel(
    const int N,
    const int64_t* __restrict__ keys,        // original keys[N]
    const int64_t* __restrict__ sorted_keys, // sorted_keys[N]
    const int64_t* __restrict__ order,       // sorted_keys[p] = keys[order[p]]
    int32_t* __restrict__ nbr                // nbr[N, 6]
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int4 coord = inverse_order_code(keys[i]);
    const int batch = coord.x, x = coord.y, y = coord.z, z = coord.w;

    // Compute key range for this batch
    const int64_t blo = static_cast<int64_t>(static_cast<uint64_t>(batch    ) << 36);
    const int64_t bhi = static_cast<int64_t>(static_cast<uint64_t>(batch + 1) << 36);

    // Find [lo, hi) in sorted_keys where keys belong to this batch
    int lo, hi;
    {
        int l = 0, r = N;
        while (l < r) {
            const int mid = (l + r) >> 1;
            if (__ldg(&sorted_keys[mid]) < blo) l = mid + 1;
            else r = mid;
        }
        lo = l;
    }
    {
        int l = 0, r = N;
        while (l < r) {
            const int mid = (l + r) >> 1;
            if (__ldg(&sorted_keys[mid]) < bhi) l = mid + 1;
            else r = mid;
        }
        hi = l;
    }

    #pragma unroll
    for (int d = 0; d < 6; d++) {
        const int nx = x + DIRS6_X[d];
        const int ny = y + DIRS6_Y[d];
        const int nz = z + DIRS6_Z[d];

        int found = -1;
        if (nx >= 0 && ny >= 0 && nz >= 0 && nx < 4096 && ny < 4096 && nz < 4096) {
            const int64_t nkey = get_order_code(make_int4(batch, nx, ny, nz));
            int l = lo, r = hi;
            while (l < r) {
                const int mid = (l + r) >> 1;
                if (__ldg(&sorted_keys[mid]) < nkey) l = mid + 1;
                else r = mid;
            }
            if (l < hi && __ldg(&sorted_keys[l]) == nkey)
                found = static_cast<int32_t>(__ldg(&order[l]));
        }
        nbr[i * 6 + d] = found;
    }
}

__global__ void dsu_init_kernel(
    const int N,
    int32_t* __restrict__ parent,
    int32_t* __restrict__ sz
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    parent[i] = i;
    sz[i] = 1;
}

__global__ void dsu_compress_kernel(
    const int N,
    int32_t* __restrict__ parent
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dsu_find(i, parent);
}

// =====================================================================
// Connected Components — specific kernels
// =====================================================================

// Single union pass: every voxel unions with all its 6-neighbors
__global__ void dsu_union_6nbr_kernel(
    const int N,
    const int32_t* __restrict__ nbr,
    int32_t* __restrict__ parent
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    #pragma unroll
    for (int d = 0; d < 6; d++) {
        const int j = nbr[i * 6 + d];
        if (j >= 0) dsu_union_basic(i, j, parent);
    }
}


// =====================================================================
// Hierarchical Patching — specific kernels
// =====================================================================

// Sort key: (batch<<48) | (code & 0x0000FFFFFFFFFFFF)
__global__ void build_sort_key_kernel(
    const int N,
    const int4* __restrict__ pos,
    const int64_t* __restrict__ code,
    int64_t* __restrict__ sort_key
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int64_t batch = static_cast<int64_t>(pos[i].x);
    sort_key[i] = (batch << 48) | (code[i] & static_cast<int64_t>(0x0000FFFFFFFFFFFFLL));
}

// Extract batch index for each position in code-sorted order
__global__ void extract_sorted_batch_kernel(
    const int N,
    const int4* __restrict__ pos,
    const int64_t* __restrict__ order,
    int32_t* __restrict__ sorted_batch
) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;
    sorted_batch[p] = pos[order[p]].x;
}

// Mark batch start positions in code-sorted order; write sentinel batch_offsets[B]=N
__global__ void build_batch_offsets_kernel(
    const int N,
    const int B,
    const int32_t* __restrict__ sorted_batch,
    int32_t* __restrict__ batch_offsets
) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;
    const int b = sorted_batch[p];
    if (p == 0 || sorted_batch[p - 1] != b)
        batch_offsets[b] = p;
    if (p == N - 1)
        batch_offsets[B] = N;
}

// Union neighbors within the same code-order segment of size L (lock-based)
__global__ void dsu_union_level_kernel(
    const int N,
    const int32_t* __restrict__ nbr,
    const int32_t* __restrict__ pos_in_sorted,
    const int32_t* __restrict__ batch_offsets,
    const int4* __restrict__ pos,
    int32_t* __restrict__ parent,
    int32_t* __restrict__ sz,
    int32_t* __restrict__ lock,
    const int L,
    const int max_patch_size
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int batch_i = pos[i].x;
    const int rank_i  = pos_in_sorted[i] - batch_offsets[batch_i];
    const int seg_i   = rank_i / L;

    #pragma unroll
    for (int d = 0; d < 6; d++) {
        const int j = nbr[i * 6 + d];
        if (j < 0) continue;
        const int batch_j = pos[j].x;
        if (batch_j != batch_i) continue;
        const int rank_j = pos_in_sorted[j] - batch_offsets[batch_j];
        const int seg_j  = rank_j / L;
        if (seg_j == seg_i)
            dsu_union_capped(i, j, parent, sz, lock, max_patch_size);
    }
}

// Attach stragglers by merging with prev/next voxel in code order (locality-preserving)
__global__ void dsu_attach_small_kernel(
    const int N,
    const int32_t* __restrict__ nbr,
    const int64_t* __restrict__ code_order,
    const int32_t* __restrict__ pos_in_sorted,
    const int4* __restrict__ pos,
    int32_t* __restrict__ parent,
    int32_t* __restrict__ sz,
    int32_t* __restrict__ lock,
    const int attach_cap
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int batch_i = pos[i].x;
    const int rank = pos_in_sorted[i];

    #pragma unroll
    for (int d = 0; d < 6; d++) {
        const int j = nbr[i * 6 + d];
        if (j < 0) continue;
        dsu_union_capped(i, j, parent, sz, lock, attach_cap);
    } 

    // Try merge with previous in code order (same batch)
    if (rank > 0) {
        const int j = static_cast<int>(code_order[rank - 1]);
        if (pos[j].x == batch_i)
            dsu_union_capped(i, j, parent, sz, lock, attach_cap);
    }

    // Try merge with next in code order (same batch)
    if (rank + 1 < N) {
        const int j = static_cast<int>(code_order[rank + 1]);
        if (pos[j].x == batch_i)
            dsu_union_capped(i, j, parent, sz, lock, attach_cap);
    }
}


// =====================================================================
// Host functions
// =====================================================================

torch::Tensor vox_connected_components_cuda(torch::Tensor pos) {
    const int N = static_cast<int>(pos.size(0));
    const auto device = pos.device();
    const auto opts32 = torch::TensorOptions().device(device).dtype(torch::kInt32);
    const auto opts64 = torch::TensorOptions().device(device).dtype(torch::kInt64);

    if (N == 0)
        return torch::empty({0}, opts32);

    const int threads = 512;
    const int blocks  = (N + threads - 1) / threads;

    // 1. Encode spatial keys
    auto keys = torch::empty({N}, opts64);
    encode_keys_kernel<<<blocks, threads>>>(
        N, reinterpret_cast<const int4*>(pos.data_ptr<int32_t>()), keys.data_ptr<int64_t>());

    // 2. Sort keys
    auto order       = torch::argsort(keys);
    auto sorted_keys = keys.index({order});

    // 3. Build 6-neighbor table
    auto nbr = torch::empty({N, 6}, opts32);
    build_6nbr_kernel<<<blocks, threads>>>(
        N,
        keys.data_ptr<int64_t>(),
        sorted_keys.data_ptr<int64_t>(),
        order.data_ptr<int64_t>(),
        nbr.data_ptr<int32_t>());

    // 4. Init DSU
    auto parent = torch::empty({N}, opts32);
    auto sz     = torch::empty({N}, opts32);
    dsu_init_kernel<<<blocks, threads>>>(N, parent.data_ptr<int32_t>(), sz.data_ptr<int32_t>());

    // 5. Single union pass — every edge visited by at least one endpoint
    auto* par     = parent.data_ptr<int32_t>();
    auto* nbr_ptr = nbr.data_ptr<int32_t>();
    dsu_union_6nbr_kernel<<<blocks, threads>>>(N, nbr_ptr, par);

    // 6. Single compress pass — flatten all paths to roots
    dsu_compress_kernel<<<blocks, threads>>>(N, par);

    return parent;
}

torch::Tensor vox_hierarchical_patches_cuda(
    torch::Tensor pos, torch::Tensor code, int max_patch_size)
{
    const int N = static_cast<int>(pos.size(0));
    const auto device = pos.device();
    const auto opts32 = torch::TensorOptions().device(device).dtype(torch::kInt32);
    const auto opts64 = torch::TensorOptions().device(device).dtype(torch::kInt64);

    if (N == 0)
        return torch::empty({0}, opts32);

    const int threads = 512;
    const int blocks  = (N + threads - 1) / threads;
    const int4* pos4  = reinterpret_cast<const int4*>(pos.data_ptr<int32_t>());

    // 1-2. Spatial keys for neighbor lookup
    auto spatial_keys = torch::empty({N}, opts64);
    encode_keys_kernel<<<blocks, threads>>>(N, pos4, spatial_keys.data_ptr<int64_t>());
    auto spatial_order       = torch::argsort(spatial_keys);
    auto spatial_sorted_keys = spatial_keys.index({spatial_order});

    // 3. Build 6-neighbor table using spatial sort
    auto nbr = torch::empty({N, 6}, opts32);
    build_6nbr_kernel<<<blocks, threads>>>(
        N,
        spatial_keys.data_ptr<int64_t>(),
        spatial_sorted_keys.data_ptr<int64_t>(),
        spatial_order.data_ptr<int64_t>(),
        nbr.data_ptr<int32_t>());

    // 4. Sort keys by (batch, code)
    auto sort_key = torch::empty({N}, opts64);
    build_sort_key_kernel<<<blocks, threads>>>(N, pos4, code.data_ptr<int64_t>(), sort_key.data_ptr<int64_t>());

    // 5. Sort by code, invert permutation
    auto code_order      = torch::argsort(sort_key);
    auto pos_in_sorted   = torch::empty({N}, opts32);
    invert_perm_kernel<<<blocks, threads>>>(N, code_order.data_ptr<int64_t>(), pos_in_sorted.data_ptr<int32_t>());

    // 6. Extract batch index in code-sorted order
    auto sorted_batch = torch::empty({N}, opts32);
    extract_sorted_batch_kernel<<<blocks, threads>>>(
        N, pos4, code_order.data_ptr<int64_t>(), sorted_batch.data_ptr<int32_t>());

    // 7. Compute B and batch_offsets[B+1]
    const int B = pos.select(1, 0).max().item<int32_t>() + 1;
    auto batch_offsets = torch::zeros({B + 1}, opts32);
    build_batch_offsets_kernel<<<blocks, threads>>>(
        N, B, sorted_batch.data_ptr<int32_t>(), batch_offsets.data_ptr<int32_t>());

    // 8. Init DSU + lock array
    auto parent = torch::empty({N}, opts32);
    auto sz     = torch::empty({N}, opts32);
    auto lock   = torch::zeros({N}, opts32);  // 0 = unlocked
    dsu_init_kernel<<<blocks, threads>>>(N, parent.data_ptr<int32_t>(), sz.data_ptr<int32_t>());

    // 9. Compute max batch length on CPU from batch_offsets
    auto offsets_cpu = batch_offsets.cpu();
    const int32_t* off_ptr = offsets_cpu.data_ptr<int32_t>();
    int max_len = 1;
    for (int b = 0; b < B; b++) {
        const int len = off_ptr[b + 1] - off_ptr[b];
        if (len > 0 && len > max_len) max_len = len;
    }

    // 10. Hierarchical union loop: L doubles from 256 until L >= max_len * 2
    int L = 256;
    auto* par       = parent.data_ptr<int32_t>();
    auto* sz_ptr    = sz.data_ptr<int32_t>();
    auto* lock_ptr  = lock.data_ptr<int32_t>();
    auto* nbr_ptr   = nbr.data_ptr<int32_t>();
    auto* pis_ptr   = pos_in_sorted.data_ptr<int32_t>();
    auto* bo_ptr    = batch_offsets.data_ptr<int32_t>();

    const int attach_cap = max_patch_size + max_patch_size / 4;  // 1.25x for straggler absorption

    do {
        dsu_union_level_kernel<<<blocks, threads>>>(
            N, nbr_ptr, pis_ptr, bo_ptr, pos4, par, sz_ptr, lock_ptr, L, max_patch_size);
        L *= 2;
    } while (L < max_len * 2);

    // 11. Attach stragglers via code-order neighbors
    auto* co_ptr = code_order.data_ptr<int64_t>();
    dsu_attach_small_kernel<<<blocks, threads>>>(
        N, nbr_ptr, co_ptr, pis_ptr, pos4, par, sz_ptr, lock_ptr, attach_cap);

    // 12. Final path compression
    dsu_compress_kernel<<<blocks, threads>>>(N, par);

    return parent;
}
