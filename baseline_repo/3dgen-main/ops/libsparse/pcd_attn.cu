#include <vector>

#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutil_math.h>

#include <bvh.h>
#include <pcd_attn.h>

// Legacy-only backward mode (point-parallel path disabled).

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define FULL_MASK 0xFFFFFFFFu

// Feature Encoding constants
#define NUM_ABS_POS 3
#define NUM_ABS_FOURIER 54
#define NUM_ABS (NUM_ABS_POS + NUM_ABS_FOURIER)  // 57
#define NUM_REL_POS 3
#define NUM_REL_FOURIER 30
#define NUM_DIST 2
#define NUM_REL (NUM_REL_POS + NUM_REL_FOURIER + NUM_DIST)  // 35
#define NUM_EXTRAS 4
#define NUM_TOTAL (NUM_ABS + NUM_REL + NUM_EXTRAS)  // 96

#define MODE_FULL 96
#define MODE_ABS 61   // 57 abs + 4 extras
#define MODE_REL 39   // 35 rel + 4 extras

constexpr size_t kDefaultMaxSmemBytes = 48 * 1024;

template <typename KernelT>
__host__ __forceinline__ void maybe_set_smem(KernelT kernel, size_t bytes) {
  if (bytes <= kDefaultMaxSmemBytes) return;
  const cudaError_t err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
  TORCH_CHECK(err == cudaSuccess,
      "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed: ",
      cudaGetErrorString(err));
}

__host__ __device__ __forceinline__ size_t align16(size_t x) {
  return (x + 15) & ~size_t(15);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(FULL_MASK, v, off);
  return v;
}

enum BiasKind { BIAS_Q = 0, BIAS_K = 1, BIAS_V = 2, BIAS_COUNT = 3 };

#ifndef FMA_INLINE
#define FMA_INLINE __forceinline__
#endif

#ifndef TK_CHUNK
#define TK_CHUNK 4
#endif

#ifndef FWD_KTILE
#define FWD_KTILE 4
#endif

#ifndef BWD_KTILE
#define BWD_KTILE 16
#endif

#ifndef F_TILE
#define F_TILE 8
#endif

#ifndef PCD_BWD_DISABLE_PVF
#define PCD_BWD_DISABLE_PVF 0
#endif

#ifndef PCD_BWD_DISABLE_PKF
#define PCD_BWD_DISABLE_PKF 0
#endif

#ifndef PCD_BWD_DISABLE_GRADW
#define PCD_BWD_DISABLE_GRADW 0
#endif

#ifndef PCD_BWD_DISABLE_BIAS
#define PCD_BWD_DISABLE_BIAS 0
#endif

#ifndef PCD_BWD_DISABLE_PROJ_QK
#define PCD_BWD_DISABLE_PROJ_QK 0
#endif

#ifndef PCD_BWD_DISABLE_PROJ_V
#define PCD_BWD_DISABLE_PROJ_V 0
#endif

#ifndef PCD_BWD_DISABLE_GW_Q
#define PCD_BWD_DISABLE_GW_Q 0
#endif

#ifndef PCD_BWD_DISABLE_GW_K
#define PCD_BWD_DISABLE_GW_K 0
#endif

#ifndef PCD_BWD_DISABLE_GW_V
#define PCD_BWD_DISABLE_GW_V 0
#endif

using smem_feat_t = float;
using smem_proj_t = half;

static constexpr size_t smem_feat_sizeof() {
    return sizeof(float);
}

static constexpr size_t smem_proj_sizeof() {
    return sizeof(half);
}

__device__ __forceinline__ float half_to_float(half v) {
    return __half2float(v);
}

__device__ __forceinline__ half float_to_half(float v) {
    return __float2half(v);
}

template <int HEAD_DIM, int N_FEATURES, bool IS_FULL, bool IS_ABS, bool IS_REL>
__device__ __forceinline__ void compute_q_bias_proj(
    int lane,
    const float* __restrict__ w_ptrs[3],
    smem_proj_t* __restrict__ q_proj_ptrs[3],
    const smem_feat_t* __restrict__ q_feat
) {
    // Q (p=0): project keypoint's static features (abs + extras, no rel).
    // K (p=1) and V (p=2): start from zero; point features are added later
    // per tile in fma_bias_proj_tile_unified so that each neighbour gets its
    // own absolute position rather than the keypoint's absolute position.
    constexpr int LANES = (HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE;

    for (int p = 0; p < 3; ++p) {
        if (!w_ptrs[p]) continue;
        smem_proj_t* proj = q_proj_ptrs[p];
        if (!proj) continue;

        if (p == BIAS_Q) {
            const float* w_base = w_ptrs[p];
            for (int li = 0; li < LANES; ++li) {
                const int dd = lane + li * WARP_SIZE;
                if (dd >= HEAD_DIM) continue;
                float sum = 0.0f;

                auto accum_f = [&](int f) {
                    const float feat_val = static_cast<float>(q_feat[f]);
                    sum += w_base[f * HEAD_DIM + dd] * feat_val;
                };

                if constexpr (IS_ABS) {
                    for (int f = 0; f < N_FEATURES; ++f) accum_f(f);
                } else if constexpr (IS_FULL) {
                    for (int f = 0; f < NUM_ABS; ++f) accum_f(f);
                    for (int f = NUM_ABS + NUM_REL; f < N_FEATURES; ++f) accum_f(f);
                } else if constexpr (IS_REL) {
                    for (int f = NUM_REL; f < N_FEATURES; ++f) accum_f(f);
                }
                proj[dd] = float_to_half(sum);
            }
        } else {
            // K and V: zero-initialise; all features come from fma_bias_proj_tile_unified
            for (int li = 0; li < LANES; ++li) {
                const int dd = lane + li * WARP_SIZE;
                if (dd < HEAD_DIM) proj[dd] = float_to_half(0.f);
            }
        }
    }
}

template <int HEAD_DIM, int N_FEATURES, bool IS_FULL, bool IS_ABS, bool IS_REL, int KTILE>
__device__ FMA_INLINE void fma_bias_proj_tile_unified(
    int lane, int kt,
    const float* __restrict__ w_ptrs[3],
    smem_proj_t* __restrict__ proj_ptrs[3],
    const smem_proj_t* __restrict__ q_proj[3],
    const smem_feat_t* __restrict__ k_feat_w
) {
    constexpr int TK_CHUNK_VAL = TK_CHUNK;
    static_assert(TK_CHUNK_VAL > 0, "TK_CHUNK must be > 0");
    constexpr int LANES = (HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE;

    for (int p = 0; p < 3; ++p) {
        smem_proj_t* proj = proj_ptrs[p];
        if (!w_ptrs[p] || !proj) continue;
        const smem_proj_t* qp = q_proj[p];
        for (int tk = 0; tk < kt; ++tk) {
            const int base = tk * HEAD_DIM;
            for (int li = 0; li < LANES; ++li) {
                const int dd = lane + li * WARP_SIZE;
                if (dd >= HEAD_DIM) continue;
                proj[base + dd] = qp ? qp[dd] : float_to_half(0.f);
            }
        }
    }
    __syncwarp();

    for (int p = 0; p < 3; ++p) {
        const float* w_base = w_ptrs[p];
        smem_proj_t* proj = proj_ptrs[p];
        if (!w_base || !proj) continue;

        for (int tk0 = 0; tk0 < kt; tk0 += TK_CHUNK_VAL) {
            float sums[LANES][TK_CHUNK_VAL];
            const int tk_max = min(TK_CHUNK_VAL, kt - tk0);
            for (int li = 0; li < LANES; ++li) {
#pragma unroll
                for (int tk = 0; tk < TK_CHUNK_VAL; ++tk) sums[li][tk] = 0.f;
            }

            auto accum_f = [&](int f) {
                for (int li = 0; li < LANES; ++li) {
                    const int dd = lane + li * WARP_SIZE;
                    if (dd >= HEAD_DIM) continue;
                    const float wv = w_base[f * HEAD_DIM + dd];
                    for (int tk = 0; tk < tk_max; ++tk) {
                        const float feat = static_cast<float>(k_feat_w[f * KTILE + (tk0 + tk)]);
                        sums[li][tk] += wv * feat;
                    }
                }
            };

            // Q (p=0): add only the neighbour-relative features so that
            //   bias_Q = W_q * [abs_keypoint, rel, extras_q(zeros)]
            // K/V (p=1,2): add ALL point features so that
            //   bias_K = W_k * [abs_point, rel, extras_point]  (FULL)
            //   bias_K = W_k * [abs_point, extras_point]        (ABS)
            //   bias_K = W_k * [rel, extras_point]              (REL)
            if constexpr (IS_FULL) {
                if (p == BIAS_Q) {
                    for (int f = NUM_ABS; f < NUM_ABS + NUM_REL; ++f) accum_f(f);
                } else {
                    for (int f = 0; f < N_FEATURES; ++f) accum_f(f);
                }
            } else if constexpr (IS_ABS) {
                if (p != BIAS_Q) {
                    for (int f = 0; f < N_FEATURES; ++f) accum_f(f);
                }
                // Q gets nothing from the tile in ABS mode; its abs features
                // are already baked in by compute_q_bias_proj.
            } else if constexpr (IS_REL) {
                if (p == BIAS_Q) {
                    for (int f = 0; f < NUM_REL; ++f) accum_f(f);
                } else {
                    for (int f = 0; f < N_FEATURES; ++f) accum_f(f);
                }
            }

            for (int tk = 0; tk < tk_max; ++tk) {
                const int tki = tk0 + tk;
                const int base = tki * HEAD_DIM;
                for (int li = 0; li < LANES; ++li) {
                    const int dd = lane + li * WARP_SIZE;
                    if (dd >= HEAD_DIM) continue;
                    proj[base + dd] = float_to_half(half_to_float(proj[base + dd]) + sums[li][tk]);
                }
            }
        }
    }
}

template <int HEAD_DIM, int N_FEATURES, bool IS_FULL, bool IS_ABS, bool IS_REL, int BIAS_KIND, int KTILE>
__device__ __forceinline__ void fma_bias_proj_tile_single_split_tk(
    int lane,
    int warp_id,
    int warp_count,
    int kt,
    const float* __restrict__ w_base,
    smem_proj_t* __restrict__ proj,
    const smem_proj_t* __restrict__ q_proj,
    const smem_feat_t* __restrict__ k_feat_w
) {
    if (!w_base || !proj) return;
    constexpr int LANES = (HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE;

    for (int tk = warp_id; tk < kt; tk += warp_count) {
        const int base = tk * HEAD_DIM;
        for (int li = 0; li < LANES; ++li) {
            const int dd = lane + li * WARP_SIZE;
            if (dd >= HEAD_DIM) continue;
            proj[base + dd] = q_proj ? q_proj[dd] : float_to_half(0.f);
        }
    }
    __syncwarp();

    for (int tk = warp_id; tk < kt; tk += warp_count) {
        float sums[LANES];
        for (int li = 0; li < LANES; ++li) sums[li] = 0.f;

        auto accum_f = [&](int f) {
            const float feat = static_cast<float>(k_feat_w[f * KTILE + tk]);
            for (int li = 0; li < LANES; ++li) {
                const int dd = lane + li * WARP_SIZE;
                if (dd >= HEAD_DIM) continue;
                sums[li] += w_base[f * HEAD_DIM + dd] * feat;
            }
        };

        if constexpr (IS_FULL) {
            if constexpr (BIAS_KIND == BIAS_Q) {
                for (int f = NUM_ABS; f < NUM_ABS + NUM_REL; ++f) accum_f(f);
            } else {
                for (int f = 0; f < N_FEATURES; ++f) accum_f(f);
            }
        } else if constexpr (IS_ABS) {
            if constexpr (BIAS_KIND != BIAS_Q) {
                for (int f = 0; f < N_FEATURES; ++f) accum_f(f);
            }
        } else if constexpr (IS_REL) {
            if constexpr (BIAS_KIND == BIAS_Q) {
                for (int f = 0; f < NUM_REL; ++f) accum_f(f);
            } else {
                for (int f = 0; f < N_FEATURES; ++f) accum_f(f);
            }
        }

        const int base = tk * HEAD_DIM;
        for (int li = 0; li < LANES; ++li) {
            const int dd = lane + li * WARP_SIZE;
            if (dd >= HEAD_DIM) continue;
            proj[base + dd] = float_to_half(half_to_float(proj[base + dd]) + sums[li]);
        }
    }
}

template <int HEAD_DIM, int N_FEATURES, int KTILE>
__device__ __forceinline__ float compute_point_bias_proj_dim(
    int dd,
    int tk,
    const float* __restrict__ w_base,
    const smem_feat_t* __restrict__ k_feat_w
) {
    if (!w_base || dd >= HEAD_DIM) return 0.0f;
    float sum = 0.0f;
#pragma unroll
    for (int f = 0; f < N_FEATURES; ++f) {
        sum += w_base[f * HEAD_DIM + dd] * static_cast<float>(k_feat_w[f * KTILE + tk]);
    }
    return sum;
}

// Deduplicated into unified version above
// __device__ __forceinline__ void fma_bias_proj_tile_bwd_feat_major


// Compute single absolute feature (position + Fourier)
// f in [0, 56]
__device__ __forceinline__ float compute_abs_feature(int f, float3 pos) {
  if (f < 3) {
    return (f == 0) ? pos.x : ((f == 1) ? pos.y : pos.z);
  }
  int loc = f - 3;
  float val = (loc < 18) ? pos.x : ((loc < 36) ? pos.y : pos.z);
  float scale = static_cast<float>(1 << ((loc % 18) / 2));  // 2^k: 1..256
  // Use accurate sin/cos (tests compare against torch.sin/cos).
  return static_cast<bool>(loc % 2) ? cosf(val * scale) : sinf(val * scale);
}

__device__ __forceinline__ float smooth_rel_clamp_value(float x, float clamp_max) {
    if (clamp_max <= 0.0f) return x;

    const float shift = 0.1f * clamp_max;
    const float linear_limit = clamp_max - shift;

    const float ax = fabsf(x);
    if (ax <= linear_limit) return x;
    if (ax >= clamp_max) return copysignf(clamp_max, x);

    const float t = (ax - linear_limit) / shift;
    const float eased_t = t * (1.0f + t - t * t);
    return copysignf(linear_limit + shift * eased_t, x);
}

// Compute single relative feature (position + Fourier + distance)
// f in [0, 34] for rel-only indexing
__device__ __forceinline__ float compute_rel_feature(int f, float dx, float dy, float dz, float dist, float dist2) {
  if (f < 3) {
    return (f == 0) ? dx : ((f == 1) ? dy : dz);
  }
  if (f < 33) {

    int loc = f - 3;
    // loc in [0, 29] maps to {dx,dy,dz} in chunks of 10
    float val = (loc < 10) ? dx : ((loc < 20) ? dy : dz);
    float scale = static_cast<float>(1 << ((loc % 10) / 2));  // 2^k: 1..16
    return  static_cast<bool>(loc % 2) ? cosf(val * scale) : sinf(val * scale);
  }

  return (f == 33) ? dist : dist2;
}

template <int KTILE, bool IS_FULL, bool IS_ABS, bool IS_REL>
__device__ __forceinline__ void encode_neighbor_features(
    smem_feat_t* __restrict__ sh_k_feat,
    int t,
    float3 pp_o,
    float3 qpos_o,
    float coord_scale,
    float rel_clamp,
    const float* __restrict__ extras,
    int P_pcd
) {
    if constexpr (IS_FULL || IS_ABS) {
        sh_k_feat[0 * KTILE + t] = pp_o.x;
        sh_k_feat[1 * KTILE + t] = pp_o.y;
        sh_k_feat[2 * KTILE + t] = pp_o.z;
    }
    if constexpr (IS_FULL || IS_REL) {
        constexpr int rel_base = IS_FULL ? NUM_ABS : 0;
        const float dx = smooth_rel_clamp_value((pp_o.x - qpos_o.x) * coord_scale, rel_clamp);
        const float dy = smooth_rel_clamp_value((pp_o.y - qpos_o.y) * coord_scale, rel_clamp);
        const float dz = smooth_rel_clamp_value((pp_o.z - qpos_o.z) * coord_scale, rel_clamp);
        sh_k_feat[(rel_base + 0) * KTILE + t] = dx;
        sh_k_feat[(rel_base + 1) * KTILE + t] = dy;
        sh_k_feat[(rel_base + 2) * KTILE + t] = dz;
        const float dist2 = dx * dx + dy * dy + dz * dz;
        sh_k_feat[(rel_base + 33) * KTILE + t] = sqrtf(dist2);
        sh_k_feat[(rel_base + 34) * KTILE + t] = dist2;
    }

    constexpr int ex_base = IS_FULL ? (NUM_ABS + NUM_REL) : (IS_ABS ? NUM_ABS : NUM_REL);
    for (int ex = 0; ex < NUM_EXTRAS; ++ex) {
        sh_k_feat[(ex_base + ex) * KTILE + t] = (extras && ex < P_pcd) ? extras[ex] : 0.0f;
    }
}

template <int KTILE, bool IS_FULL, bool IS_ABS, bool IS_REL>
__device__ __forceinline__ void fill_fourier_features(
    smem_feat_t* __restrict__ sh_k_feat,
    int thread_idx,
    int thread_step,
    int kt
) {
    if constexpr (IS_FULL || IS_ABS) {
        for (int idx = thread_idx; idx < kt * NUM_ABS_POS; idx += thread_step) {
            const int t = idx / NUM_ABS_POS;
            const int cidx = idx - t * NUM_ABS_POS;
            const float val = static_cast<float>(sh_k_feat[cidx * KTILE + t]);
#pragma unroll
            for (int fidx = 0; fidx < 9; ++fidx) {
                float s, c;
                sincosf(val * static_cast<float>(1 << fidx), &s, &c);
                const int base = 3 + (cidx * 9 + fidx) * 2;
                sh_k_feat[base * KTILE + t] = s;
                sh_k_feat[(base + 1) * KTILE + t] = c;
            }
        }
    }
    if constexpr (IS_FULL || IS_REL) {
        constexpr int rel_base = IS_FULL ? NUM_ABS : 0;
        for (int idx = thread_idx; idx < kt * NUM_REL_POS; idx += thread_step) {
            const int t = idx / NUM_REL_POS;
            const int cidx = idx - t * NUM_REL_POS;
            const float val = static_cast<float>(sh_k_feat[(rel_base + cidx) * KTILE + t]);
#pragma unroll
            for (int fidx = 0; fidx < 5; ++fidx) {
                float s, c;
                sincosf(val * static_cast<float>(1 << fidx), &s, &c);
                const int base = rel_base + 3 + (cidx * 5 + fidx) * 2;
                sh_k_feat[base * KTILE + t] = s;
                sh_k_feat[(base + 1) * KTILE + t] = c;
            }
        }
    }
}

#define MAX_STACK_DEPTH 16
#define MAX_STACK_SIZE (MAX_STACK_DEPTH * 2)

__device__ __forceinline__ float bbox_dist2(const float3 p, const float3 mn, const float3 mx) {
    float dx = fmaxf(fmaxf(mn.x - p.x, 0.f), p.x - mx.x);
    float dy = fmaxf(fmaxf(mn.y - p.y, 0.f), p.y - mx.y);
    float dz = fmaxf(fmaxf(mn.z - p.z, 0.f), p.z - mx.z);
    return dx*dx + dy*dy + dz*dz;
}

__device__ __forceinline__ void heap_swap(float &a, float &b) { float t=a; a=b; b=t; }
__device__ __forceinline__ void heap_swap(int &a, int &b) { int t=a; a=b; b=t; }

__device__ __forceinline__ void heap_sift_up(float* hd2, int* hidx, int i) {
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (hd2[p] >= hd2[i]) break;
        heap_swap(hd2[p], hd2[i]);
        heap_swap(hidx[p], hidx[i]);
        i = p;
    }
}
__device__ __forceinline__ void heap_sift_down(float* hd2, int* hidx, int n, int i) {
    while (true) {
        int l = i * 2 + 1;
        if (l >= n) break;
        int r = l + 1;
        int m = (r < n && hd2[r] > hd2[l]) ? r : l;
        if (hd2[i] >= hd2[m]) break;
        heap_swap(hd2[i], hd2[m]);
        heap_swap(hidx[i], hidx[m]);
        i = m;
    }
}
__device__ __forceinline__ void heap_try_insert(float* hd2, int* hidx, int &hn, int K, float d2, int pid) {
    if (hn < K) {
        hd2[hn] = d2;
        hidx[hn] = pid;
        heap_sift_up(hd2, hidx, hn);
        hn++;
    } else {
        if (d2 >= hd2[0]) return;     // worse than current worst
        hd2[0] = d2;
        hidx[0] = pid;
        heap_sift_down(hd2, hidx, hn, 0);
    }
}

__global__ void pcd_knn_kernel(
    const int M_TOTAL,
    const int M_PER_BATCH,
    const int N_PER_BATCH,
    const int NODES_PER_BATCH,
    const int K,
    const float radius,
    const float3* __restrict__ key_pos,
    const float3* __restrict__ sorted_pts,
    const int* __restrict__ sorted_to_orig,
    const int3* __restrict__ nodes,
    const int2* __restrict__ ranges,
    const float3* __restrict__ bbox,
    int* __restrict__ nn_idx,
    int* __restrict__ nn_count
) {
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (warp_id >= M_TOTAL) return;

    const int batch_idx = warp_id / M_PER_BATCH;
    const int point_offset = batch_idx * N_PER_BATCH;
    const int node_offset = batch_idx * NODES_PER_BATCH;

    // ---- dynamic shared memory: [hd2 floats][hpid ints] ----
    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned char* smem = smem_raw;

    float* sh_d2 = reinterpret_cast<float*>(smem);
    smem += align16(static_cast<size_t>(WARPS_PER_BLOCK) * K * sizeof(float));

    int* sh_pid = reinterpret_cast<int*>(smem);
    // smem += align16(static_cast<size_t>(WARPS_PER_BLOCK) * K * sizeof(int));

    float* hd2  = sh_d2 + static_cast<size_t>(warp_in_block) * K;
    int*   hpid = sh_pid + static_cast<size_t>(warp_in_block) * K;

    // init heap (warp-parallel)
    for (int i = lane; i < K; i += WARP_SIZE) {
        hd2[i] = 1e30f; hpid[i] = -1;
    }
    __syncwarp();

    const float3 qpos = key_pos[warp_id];
    float r2_cap = (radius > 0.f) ? radius * radius : 1e30f;

    int stack_node[MAX_STACK_SIZE];
    int stack_depth[MAX_STACK_SIZE];
    int stack_size = 0;

    int heap_n = 0;
    if (lane == 0) {
        stack_node[0] = 0; stack_depth[0] = 0; stack_size++;
    }

    while (true) {
        int node = -1, depth = 0;
        if (lane == 0 && stack_size > 0) {
            --stack_size;
            node = stack_node[stack_size];
            depth = stack_depth[stack_size];
        }
        node  = __shfl_sync(FULL_MASK, node, 0);
        depth = __shfl_sync(FULL_MASK, depth, 0);
        if (node < 0) break;

        int left = -1, right = -1;
        if (lane == 0) { int3 c = nodes[node_offset + node]; left = c.x; right = c.y; }
        left  = __shfl_sync(FULL_MASK, left, 0);
        right = __shfl_sync(FULL_MASK, right, 0);

        const bool is_leaf = ((left < 0 && right < 0) || (depth >= MAX_STACK_DEPTH));

        if (is_leaf) {
            int lb = 0, rb = -1;
            if (lane == 0) { int2 b = ranges[node_offset + node]; lb = b.x; rb = b.y; }
            lb = __shfl_sync(FULL_MASK, lb, 0);
            rb = __shfl_sync(FULL_MASK, rb, 0);

            for (int base = lb; base <= rb; base += WARP_SIZE) {

                int p = base + lane;

                float d2 = 1e30f;
                if (p <= rb) {
                    const float3 pp = sorted_pts[point_offset + p];
                    float dx = pp.x - qpos.x, dy = pp.y - qpos.y, dz = pp.z - qpos.z;
                    d2 = dx*dx + dy*dy + dz*dz;
                }

                unsigned int m = __ballot_sync(FULL_MASK, static_cast<int>(p <= rb && d2 < r2_cap));

                while (m) {
                    int l = __ffs(m) - 1;
                    float d2_l = __shfl_sync(FULL_MASK, d2, l);
                    int p_l = __shfl_sync(FULL_MASK, p, l);
                    if (lane == 0) {
                        heap_try_insert(hd2, hpid, heap_n, K, d2_l, p_l);
                        if (heap_n >= K) r2_cap = fminf(r2_cap, hd2[0]);
                    }
                    m = m & (~(1u << l));
                }
                r2_cap = __shfl_sync(FULL_MASK, r2_cap, 0);
            }
        } else {
            if (lane == 0) {
                float dl = 1e30f, dr = 1e30f;
                bool travL = false, travR = false;
                if (left >= 0) {
                    float3 mn = bbox[(node_offset + left) * 2 + 0];
                    float3 mx = bbox[(node_offset + left) * 2 + 1];
                    dl = bbox_dist2(qpos, mn, mx);
                    travL = (dl < r2_cap);
                }
                if (right >= 0) {
                    float3 mn = bbox[(node_offset + right) * 2 + 0];
                    float3 mx = bbox[(node_offset + right) * 2 + 1];
                    dr = bbox_dist2(qpos, mn, mx);
                    travR = (dr < r2_cap);
                }

                if (travL && travR) {
                    if (dl < dr) {
                        stack_node[stack_size] = right; stack_depth[stack_size] = depth + 1; stack_size++;
                        stack_node[stack_size] = left;  stack_depth[stack_size] = depth + 1; stack_size++;
                    } else {
                        stack_node[stack_size] = left;  stack_depth[stack_size] = depth + 1; stack_size++;
                        stack_node[stack_size] = right; stack_depth[stack_size] = depth + 1; stack_size++;
                    }
                } else if (travL) {
                    stack_node[stack_size] = left;  stack_depth[stack_size] = depth + 1; stack_size++;
                } else if (travR) {
                    stack_node[stack_size] = right; stack_depth[stack_size] = depth + 1; stack_size++;
                }
            }
        }
    }
    __syncwarp();

    // write found neighbours
    int count = __shfl_sync(FULL_MASK, heap_n, 0);
    if (lane == 0) nn_count[warp_id] = count;
    for (int i = lane; i < K; i += WARP_SIZE) {
        if (i < count) {
            const int local_orig = sorted_to_orig[point_offset + hpid[i]];
            nn_idx[static_cast<size_t>(warp_id) * K + i] = point_offset + local_orig;
        } else {
            nn_idx[static_cast<size_t>(warp_id) * K + i] = -1;
        }
    }
}


std::vector<torch::Tensor> cuda_pcd_knn(
    torch::Tensor keypoints,
    torch::Tensor points,
    int K, float radius
) {
    const bool key_batched = (keypoints.dim() == 3);
    const bool points_batched = (points.dim() == 3);
    TORCH_CHECK(key_batched == points_batched,
        "keypoints and points must both be batched or both be unbatched");

    const int B = points_batched ? points.size(0) : 1;
    const int N = points_batched ? points.size(1) : points.size(0);
    const int M_per_batch = key_batched ? keypoints.size(1) : keypoints.size(0);
    const int M_total = B * M_per_batch;
    const int nodes_per_batch = 2 * N - 1;
    auto key_flat = key_batched ? keypoints.view({M_total, 3}) : keypoints;

    // Build BVH
    auto pcd = points_batched ? points : points.unsqueeze(0); // (B, N, 3)
    auto pcd_bbox = pcd; // Simplification, build_bvh will handle
    bvh tree = build_bvh(pcd, pcd_bbox);

    auto sorted_pts = tree.points.view({B * N, 3});
    auto sorted_to_orig = tree.orig_idx.view({B * N}).to(torch::kInt32);
    auto nodes = tree.nodes.view({B * nodes_per_batch, 3});
    auto ranges = tree.ranges.view({B * nodes_per_batch, 2});
    auto bbox = tree.bbox.view({B * nodes_per_batch, 6});

    auto nn_idx_flat = torch::full({M_total, K}, -1, torch::TensorOptions().device(keypoints.device()).dtype(torch::kInt32));
    auto nn_count_flat = torch::zeros({M_total}, torch::TensorOptions().device(keypoints.device()).dtype(torch::kInt32));

    const int threads = WARPS_PER_BLOCK * 32;
    const int blocks = (M_total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    size_t shmem =
        align16(static_cast<size_t>(WARPS_PER_BLOCK) * K * sizeof(float)) + // hd2
        align16(static_cast<size_t>(WARPS_PER_BLOCK) * K * sizeof(int));    // hpid

    pcd_knn_kernel<<<blocks, threads, shmem>>>(
        M_total, M_per_batch, N, nodes_per_batch, K, radius,
        reinterpret_cast<float3*>(key_flat.data_ptr<float>()),
        reinterpret_cast<float3*>(sorted_pts.data_ptr<float>()),
        sorted_to_orig.data_ptr<int>(),
        reinterpret_cast<int3*>(nodes.data_ptr<int>()),
        reinterpret_cast<int2*>(ranges.data_ptr<int>()),
        reinterpret_cast<float3*>(bbox.data_ptr<float>()),
        nn_idx_flat.data_ptr<int>(),
        nn_count_flat.data_ptr<int>()
    );

    if (key_batched) {
        return {
            nn_idx_flat.view({B, M_per_batch, K}),
            nn_count_flat.view({B, M_per_batch})
        };
    }
    return {nn_idx_flat, nn_count_flat};
}

__device__ __forceinline__ int clamp_i32(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ int voxel_coord_from_point01(float p01, int grid_res) {
    float p = fminf(fmaxf(p01, 0.0f), 1.0f);
    if (p >= 1.0f) p = 1.0f - 1e-7f;
    int c = static_cast<int>(floorf(p * static_cast<float>(grid_res)));
    return clamp_i32(c, 0, grid_res - 1);
}

__global__ void build_voxel_lookup_kernel(
    const float3* __restrict__ points01,
    int B,
    int N_per_batch,
    int grid_res,
    int* __restrict__ voxel_to_point
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * N_per_batch;
    if (idx >= total) return;

    const int b = idx / N_per_batch;
    const float3 p = points01[idx];
    const int ix = voxel_coord_from_point01(p.x, grid_res);
    const int iy = voxel_coord_from_point01(p.y, grid_res);
    const int iz = voxel_coord_from_point01(p.z, grid_res);

    const size_t off =
        (((static_cast<size_t>(b) * grid_res + ix) * grid_res + iy) * grid_res + iz);

    // Expected unique occupancy; keep smallest point index if duplicates appear.
    int old = atomicCAS(&voxel_to_point[off], -1, idx);
    if (old != -1 && idx < old) atomicMin(&voxel_to_point[off], idx);
}

__global__ void pcd_vox_knn_kernel(
    const float3* __restrict__ keypoints01,
    const int* __restrict__ voxel_to_point,
    int B,
    int M_per_batch,
    int grid_res,
    int dilation,
    int Kmax,
    int* __restrict__ nn_idx,
    int* __restrict__ nn_count
) {
    const int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int M_total = B * M_per_batch;
    if (key_idx >= M_total) return;

    const int b = key_idx / M_per_batch;
    const float3 kp = keypoints01[key_idx];
    const float gx = fminf(fmaxf(kp.x, 0.0f), 1.0f) * static_cast<float>(grid_res);
    const float gy = fminf(fmaxf(kp.y, 0.0f), 1.0f) * static_cast<float>(grid_res);
    const float gz = fminf(fmaxf(kp.z, 0.0f), 1.0f) * static_cast<float>(grid_res);

    const float rx = roundf(gx), ry = roundf(gy), rz = roundf(gz);
    const bool on_x = fabsf(gx - rx) <= 1e-6f, on_y = fabsf(gy - ry) <= 1e-6f, on_z = fabsf(gz - rz) <= 1e-6f;
    const int n_on = static_cast<int>(on_x) + static_cast<int>(on_y) + static_cast<int>(on_z);

    const int fx = static_cast<int>(floorf(gx)), fy = static_cast<int>(floorf(gy)), fz = static_cast<int>(floorf(gz));
    const int ix = static_cast<int>(rx), iy = static_cast<int>(ry), iz = static_cast<int>(rz);

    auto axis_range = [&](bool on_axis, int floor_i, int int_i, int* s, int* e) {
        if (on_axis) {
            if (dilation == 0) {
                *s = int_i - 1;
                *e = int_i - 1;
                return;
            }
            *s = int_i - dilation;
            *e = int_i + dilation - 1;
        } else {
            const int base = clamp_i32(floor_i, 0, grid_res - 1);
            *s = base - dilation;
            *e = base + dilation;
        }
    };

    int sx, ex, sy, ey, sz, ez;
    axis_range(on_x, fx, ix, &sx, &ex);
    axis_range(on_y, fy, iy, &sy, &ey);
    axis_range(on_z, fz, iz, &sz, &ez);

    int count = 0;
    const size_t out_row = static_cast<size_t>(key_idx) * Kmax;
    for (int x = sx; x <= ex; ++x) {
        if (x < 0 || x >= grid_res) continue;
        for (int y = sy; y <= ey; ++y) {
            if (y < 0 || y >= grid_res) continue;
            for (int z = sz; z <= ez; ++z) {
                if (z < 0 || z >= grid_res) continue;
                const size_t off =
                    (((static_cast<size_t>(b) * grid_res + x) * grid_res + y) * grid_res + z);
                const int pid = voxel_to_point[off];
                if (pid >= 0) {
                    if (count < Kmax) nn_idx[out_row + count] = pid;
                    ++count;
                }
            }
        }
    }
    nn_count[key_idx] = count > Kmax ? Kmax : count;
}

std::vector<torch::Tensor> cuda_pcd_vox_knn(
    torch::Tensor keypoints,
    torch::Tensor points,
    int grid_res,
    int dilation
) {
    const bool key_batched = (keypoints.dim() == 3);
    const bool points_batched = (points.dim() == 3);
    TORCH_CHECK(key_batched == points_batched,
        "keypoints and points must both be batched or both be unbatched");

    const int B = points_batched ? points.size(0) : 1;
    const int N_per_batch = points_batched ? points.size(1) : points.size(0);
    const int M_per_batch = key_batched ? keypoints.size(1) : keypoints.size(0);
    const int M_total = B * M_per_batch;

    auto key_flat = key_batched ? keypoints.view({M_total, 3}) : keypoints;
    auto points_flat = points_batched ? points.view({B * N_per_batch, 3}) : points;

    int Kmax = (2 * dilation + 1) * (2 * dilation + 1) * (2 * dilation + 1);
    if (Kmax < 2) Kmax = 2;  // plane d=0 needs two touching voxels.

    auto voxel_to_point = torch::full(
        {B, grid_res, grid_res, grid_res},
        -1,
        torch::TensorOptions().device(points.device()).dtype(torch::kInt32));
    auto nn_idx_flat = torch::full(
        {M_total, Kmax},
        -1,
        torch::TensorOptions().device(keypoints.device()).dtype(torch::kInt32));
    auto nn_count_flat = torch::zeros(
        {M_total},
        torch::TensorOptions().device(keypoints.device()).dtype(torch::kInt32));

    {
        const int threads = 256;
        const int total = B * N_per_batch;
        const int blocks = (total + threads - 1) / threads;
        build_voxel_lookup_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float3*>(points_flat.data_ptr<float>()),
            B,
            N_per_batch,
            grid_res,
            voxel_to_point.data_ptr<int>());
    }

    {
        const int threads = 256;
        const int blocks = (M_total + threads - 1) / threads;
        pcd_vox_knn_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float3*>(key_flat.data_ptr<float>()),
            voxel_to_point.data_ptr<int>(),
            B,
            M_per_batch,
            grid_res,
            dilation,
            Kmax,
            nn_idx_flat.data_ptr<int>(),
            nn_count_flat.data_ptr<int>());
    }

    nn_idx_flat = std::get<0>(nn_idx_flat.sort(/*dim=*/-1, /*descending=*/true));

    if (key_batched) {
        return {
            nn_idx_flat.view({B, M_per_batch, Kmax}),
            nn_count_flat.view({B, M_per_batch})
        };
    }
    return {nn_idx_flat, nn_count_flat};
}



template <int HEAD_DIM, int N_FEATURES, bool SHARE_GROUP4>
__global__ void pcd_attn_feat_fwd_kernel(
    const int M,
    const int C,
    const int K,
    const float coord_scale,
    const float rel_clamp,
    const int P_pcd,
    const float3* __restrict__ key_pos,
    const float* __restrict__ key_feat,
    const float3* __restrict__ points,
    const float* __restrict__ point_kfeat,
    const float* __restrict__ point_vfeat,
    const float* __restrict__ point_extras,
    const int* __restrict__ nn_idx,
    const int* __restrict__ nn_count,
    const float* __restrict__ weights,  // (H, 3, N_FEATURES, HEAD_DIM) in fp32
    const float* __restrict__ bias_base,  // (H, 3, HEAD_DIM) in fp32, or nullptr
    float* __restrict__ out,
    float* __restrict__ head_max,
    float* __restrict__ head_denom
) {
    constexpr bool IS_FULL = (N_FEATURES == MODE_FULL);
    constexpr bool IS_ABS  = (N_FEATURES == MODE_ABS);
    constexpr bool IS_REL  = (N_FEATURES == MODE_REL);

    constexpr int KTILE = FWD_KTILE;
    static_assert(KTILE > 0, "FWD_KTILE must be > 0");

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;   // [0, WARPS_PER_BLOCK)
    const int lane = tid & 31;      // [0, 31]
    const int key_idx = blockIdx.x;
    if (key_idx >= M) return;
    constexpr int PROJ_SLOTS = SHARE_GROUP4 ? 1 : WARPS_PER_BLOCK;

    using feat_t = smem_feat_t;
    using proj_t = smem_proj_t;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned char* smem = smem_raw;

    feat_t* sh_q_feat = reinterpret_cast<feat_t*>(smem);
    smem += align16(N_FEATURES * sizeof(feat_t));

    int* sh_pid = reinterpret_cast<int*>(smem);
    smem += align16(KTILE * sizeof(int));

    // K features: [N_FEATURES, KTILE] feat-major layout
    feat_t* sh_k_feat = reinterpret_cast<feat_t*>(smem);
    smem += align16(KTILE * N_FEATURES * sizeof(feat_t));

    // Bias projection buffers: per-warp buffers so each warp/head can run independently.
    proj_t* sh_bq_proj = reinterpret_cast<proj_t*>(smem);   // [WARPS_PER_BLOCK, KTILE, HEAD_DIM]
    smem += align16(static_cast<size_t>(PROJ_SLOTS) * KTILE * HEAD_DIM * sizeof(proj_t));
    proj_t* sh_bk_proj = reinterpret_cast<proj_t*>(smem);
    smem += align16(static_cast<size_t>(PROJ_SLOTS) * KTILE * HEAD_DIM * sizeof(proj_t));
    proj_t* sh_bv_proj = reinterpret_cast<proj_t*>(smem);
    smem += align16(static_cast<size_t>(PROJ_SLOTS) * KTILE * HEAD_DIM * sizeof(proj_t));

    // Absolute parts of the bias projection (computed once per query)
    proj_t* sh_bq_qproj = reinterpret_cast<proj_t*>(smem);
    smem += align16(static_cast<size_t>(PROJ_SLOTS) * HEAD_DIM * sizeof(proj_t));



    const int count = nn_count[key_idx];
    const int H = C / HEAD_DIM;
    const int H_proj = SHARE_GROUP4 ? (H / WARPS_PER_BLOCK) : H;

    // Handle empty case
    if (count == 0) {
        for (int c = threadIdx.x; c < C; c += blockDim.x)
            out[static_cast<size_t>(key_idx) * C + c] = 0.f;
        for (int hh = threadIdx.x; hh < H; hh += blockDim.x) {
            head_max[static_cast<size_t>(key_idx) * H + hh] = -1e30f;
            head_denom[static_cast<size_t>(key_idx) * H + hh] = 1.f;
        }
        return;
    }

    // Get key position in original [-1, 1] space
    const float3 qpos = key_pos[key_idx];
    const float3 qpos_o = make_float3(qpos.x * 2.0f - 1.0f, qpos.y * 2.0f - 1.0f, qpos.z * 2.0f - 1.0f);

    // Compute Q features (once per keypoint)
    // NOTE: Q's relative/dist features are neighbor-dependent, so here we only precompute
    // the *absolute* part for the keypoint (and keep other slots at 0).
    if constexpr (IS_FULL || IS_ABS) {
        for (int f = tid; f < N_FEATURES; f += blockDim.x) {
            float val = 0.0f;
            if (f < NUM_ABS) {
                val = compute_abs_feature(f, qpos_o);
            }
            sh_q_feat[f] = val;
        }
    } else { // IS_REL
        for (int f = tid; f < N_FEATURES; f += blockDim.x) {
            sh_q_feat[f] = 0.f;
        }
    }
    __syncthreads();

    const float* qbase = key_feat ? key_feat + static_cast<size_t>(key_idx) * C : nullptr;
    const float inv_sqrt_d = rsqrtf(static_cast<float>(HEAD_DIM));

    // One warp per head (4 heads per block), with lane-striding over HEAD_DIM.
    for (int h0 = 0; h0 < H; h0 += WARPS_PER_BLOCK) {
        const int h = h0 + warp_id;
        const int h_proj = SHARE_GROUP4 ? (h0 / WARPS_PER_BLOCK) : h;
        const bool active = (h < H);

        const float* qf = (active && qbase) ? (qbase + h * HEAD_DIM) : nullptr;

        const float* bq_h = (active && bias_base) ? bias_base + h * 3 * HEAD_DIM : nullptr;
        const float* bk_h = (active && bias_base) ? bias_base + h * 3 * HEAD_DIM + HEAD_DIM : nullptr;
        const float* bv_h = (active && bias_base) ? bias_base + h * 3 * HEAD_DIM + 2 * HEAD_DIM : nullptr;

        const float* w_base = ((h_proj < H_proj) && weights) ? weights + h_proj * 3 * N_FEATURES * HEAD_DIM : nullptr;
        const float* w_q = w_base;
        const float* w_k = w_base ? (w_base + N_FEATURES * HEAD_DIM) : nullptr;
        const float* w_v = w_base ? (w_base + 2 * N_FEATURES * HEAD_DIM) : nullptr;

        float max_logit = -1e30f;
        float sum_exp = 0.f;
        constexpr int LANES = (HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE;
        float acc[LANES] = {0.f};

        for (int k0 = 0; k0 < count; k0 += KTILE) {
            const int kt = min(KTILE, count - k0);

            // Stage PIDs first (shared for all warps)
            for (int t = tid; t < kt; t += blockDim.x) {
                int pid = nn_idx[static_cast<size_t>(key_idx) * K + (k0 + t)];
                sh_pid[t] = pid;
            }
            __syncthreads();

            // Encode features into [N_FEATURES, KTILE] feat-major layout.
            for (int t = tid; t < kt; t += blockDim.x) {
                const int pid = sh_pid[t];
                const float3 pp = points[pid];
                    const float3 pp_o = make_float3(pp.x * 2.0f - 1.0f, pp.y * 2.0f - 1.0f, pp.z * 2.0f - 1.0f);
                    const float* extras = (point_extras && P_pcd > 0) ? point_extras + static_cast<size_t>(pid) * P_pcd : nullptr;
                    encode_neighbor_features<KTILE, IS_FULL, IS_ABS, IS_REL>(
                        sh_k_feat, t, pp_o, qpos_o, coord_scale, rel_clamp, extras, P_pcd);
                }
            __syncthreads();

            fill_fourier_features<KTILE, IS_FULL, IS_ABS, IS_REL>(sh_k_feat, tid, blockDim.x, kt);
            __syncthreads();

            const int proj_warp = SHARE_GROUP4 ? 0 : warp_id;
            proj_t* bq_proj_w = sh_bq_proj + static_cast<size_t>(proj_warp) * KTILE * HEAD_DIM;
            proj_t* bk_proj_w = sh_bk_proj + static_cast<size_t>(proj_warp) * KTILE * HEAD_DIM;
            proj_t* bv_proj_w = sh_bv_proj + static_cast<size_t>(proj_warp) * KTILE * HEAD_DIM;
            const bool has_proj = (w_q || w_k || w_v);

            proj_t* qp_q = sh_bq_qproj + static_cast<size_t>(proj_warp) * HEAD_DIM;

            if (has_proj) {
                if constexpr (SHARE_GROUP4) {
                    if (k0 == 0) {
                        if (warp_id == 0) {
                            const float* w_ptrs_q[BIAS_COUNT] = {w_q, nullptr, nullptr};
                            proj_t* q_proj_ptrs_q[BIAS_COUNT] = {qp_q, nullptr, nullptr};
                            compute_q_bias_proj<HEAD_DIM, N_FEATURES, IS_FULL, IS_ABS, IS_REL>(
                                lane, w_ptrs_q, q_proj_ptrs_q, sh_q_feat);
                        }
                    }
                    __syncthreads();

                    fma_bias_proj_tile_single_split_tk<HEAD_DIM, N_FEATURES, IS_FULL, IS_ABS, IS_REL, BIAS_Q, KTILE>(
                        lane, warp_id, WARPS_PER_BLOCK, kt, w_q, bq_proj_w, qp_q, sh_k_feat);
                    fma_bias_proj_tile_single_split_tk<HEAD_DIM, N_FEATURES, IS_FULL, IS_ABS, IS_REL, BIAS_K, KTILE>(
                        lane, warp_id, WARPS_PER_BLOCK, kt, w_k, bk_proj_w, nullptr, sh_k_feat);
                    fma_bias_proj_tile_single_split_tk<HEAD_DIM, N_FEATURES, IS_FULL, IS_ABS, IS_REL, BIAS_V, KTILE>(
                        lane, warp_id, WARPS_PER_BLOCK, kt, w_v, bv_proj_w, nullptr, sh_k_feat);
                } else {
                    const float* w_ptrs[BIAS_COUNT] = {w_q, w_k, w_v};
                    proj_t* proj_ptrs[BIAS_COUNT] = {bq_proj_w, bk_proj_w, bv_proj_w};
                    if (k0 == 0) {
                        proj_t* n_q_proj_ptrs[BIAS_COUNT] = {qp_q, nullptr, nullptr};
                        compute_q_bias_proj<HEAD_DIM, N_FEATURES, IS_FULL, IS_ABS, IS_REL>(
                            lane, w_ptrs, n_q_proj_ptrs, sh_q_feat);
                        __syncwarp();
                    }

                    const proj_t* const_q_proj_ptrs[BIAS_COUNT] = {qp_q, nullptr, nullptr};
                    fma_bias_proj_tile_unified<HEAD_DIM, N_FEATURES, IS_FULL, IS_ABS, IS_REL, KTILE>(
                        lane, kt, w_ptrs, proj_ptrs, const_q_proj_ptrs, sh_k_feat);
                    __syncwarp();
                }
            }

            // Make sure all warps are done with this tile's shared buffers before overwriting in next tile.
            __syncthreads();

            if (active) {
                for (int tk = 0; tk < kt; ++tk) {
                    const int pid = sh_pid[tk];

                    const float* kf_ptr = point_kfeat ?
                        point_kfeat + (static_cast<size_t>(pid) * C + h * HEAD_DIM) : nullptr;
                    const float* vf_ptr = point_vfeat ?
                        point_vfeat + (static_cast<size_t>(pid) * C + h * HEAD_DIM) : nullptr;

                    float dot = 0.f;
                    float v_eff[LANES];

#pragma unroll
                    for (int li = 0; li < LANES; ++li) {
                        const int dd = lane + li * WARP_SIZE;
                        if (dd >= HEAD_DIM) { v_eff[li] = 0.f; continue; }

                        float qe = qf ? qf[dd] : 0.f;
                        if (bq_h) qe += bq_h[dd];
                        if (w_q) qe += half_to_float(bq_proj_w[tk * HEAD_DIM + dd]);

                        float ke = kf_ptr ? kf_ptr[dd] : 0.f;
                        if (bk_h) ke += bk_h[dd];
                        if (w_k) ke += half_to_float(bk_proj_w[tk * HEAD_DIM + dd]);

                        dot += qe * ke;

                        float ve = vf_ptr ? vf_ptr[dd] : 0.f;
                        if (bv_h) ve += bv_h[dd];
                        if (w_v) ve += half_to_float(bv_proj_w[tk * HEAD_DIM + dd]);
                        v_eff[li] = ve;
                    }
                    dot = warp_reduce_sum(dot);
                    float logit = dot * inv_sqrt_d;

                    float alpha = 1.f, p = 0.f;
                    if (lane == 0) {
                        float max_new = fmaxf(max_logit, logit);
                        alpha = expf(max_logit - max_new);
                        p = expf(logit - max_new);
                        sum_exp = sum_exp * alpha + p;
                        max_logit = max_new;
                    }
                    alpha = __shfl_sync(FULL_MASK, alpha, 0);
                    p = __shfl_sync(FULL_MASK, p, 0);
                    sum_exp = __shfl_sync(FULL_MASK, sum_exp, 0);

#pragma unroll
                    for (int li = 0; li < LANES; ++li) {
                        if ((lane + li * WARP_SIZE) < HEAD_DIM)
                            acc[li] = acc[li] * alpha + p * v_eff[li];
                    }
                }
            }
            __syncthreads();
        }

        sum_exp = fmaxf(sum_exp, 1e-12f);
        if (active) {
            if (lane == 0) {
                head_max[static_cast<size_t>(key_idx) * H + h] = max_logit;
                head_denom[static_cast<size_t>(key_idx) * H + h] = sum_exp;
            }

            float* outp = out + (static_cast<size_t>(key_idx) * C + h * HEAD_DIM);
            const float inv_sum = 1.0f / sum_exp;
            for (int li = 0; li < LANES; ++li) {
                const int dd = lane + li * WARP_SIZE;
                if (dd < HEAD_DIM) outp[dd] = acc[li] * inv_sum;
            }
        }

        __syncthreads();
    }
}


// Feature-mode launcher helper
template <int HEAD_DIM>
static void launch_attn_feat_fwd(
    int M, int C, int K, float coord_scale, float rel_clamp, int proj_share_group,
    torch::Tensor keypoints, torch::Tensor key_feat,
    torch::Tensor points, torch::Tensor point_kfeat, torch::Tensor point_vfeat,
    torch::Tensor point_extras,
    torch::Tensor nn_idx, torch::Tensor nn_count,
    torch::Tensor weights, torch::Tensor bias,
    torch::Tensor out, torch::Tensor head_max, torch::Tensor head_denom,
    int N_FEATURES
) {
    const int threads = WARPS_PER_BLOCK * 32;
    const int blocks = M;

    constexpr int KTILE = FWD_KTILE;
    static_assert(KTILE > 0, "FWD_KTILE must be > 0");
    int P_pcd = (point_extras.defined() && point_extras.numel() > 0) ? point_extras.size(1) : 0;

    // Compute shared memory size (per-kernel specialization)
    auto shmem_base = [&](int nf, bool share_group4) -> size_t {
        const int proj_slots = share_group4 ? 1 : WARPS_PER_BLOCK;
        const size_t feat_sz = smem_feat_sizeof();
        const size_t proj_sz = smem_proj_sizeof();
        size_t base = align16(nf * feat_sz)              // Q features
                    + align16(KTILE * sizeof(int))        // PIDs
                    + align16(KTILE * nf * feat_sz);      // K features [N_FEATURES, KTILE]

        base += align16(static_cast<size_t>(proj_slots) * KTILE * HEAD_DIM * proj_sz) * 3; // bq/bk/bv proj
        base += align16(static_cast<size_t>(proj_slots) * HEAD_DIM * proj_sz); // q proj base (q only)
        return base;
    };

    const float* key_feat_ptr = (key_feat.defined() && key_feat.numel() > 0) ?
        key_feat.data_ptr<float>() : nullptr;
    const float* point_kfeat_ptr = (point_kfeat.defined() && point_kfeat.numel() > 0) ?
        point_kfeat.data_ptr<float>() : nullptr;
    const float* point_vfeat_ptr = (point_vfeat.defined() && point_vfeat.numel() > 0) ?
        point_vfeat.data_ptr<float>() : nullptr;
    const float* point_extras_ptr = (point_extras.defined() && point_extras.numel() > 0) ?
        point_extras.data_ptr<float>() : nullptr;

    const float* weights_ptr = (weights.defined() && weights.numel() > 0) ?
        weights.data_ptr<float>() : nullptr;
    const float* bias_ptr = (bias.defined() && bias.numel() > 0) ?
        bias.data_ptr<float>() : nullptr;

    // Dispatch based on feature count and projection-sharing mode.
    #define LAUNCH_FEAT_KERNEL(NF, SHARE4) \
        do { \
            size_t shmem = shmem_base(NF, SHARE4); \
            maybe_set_smem(pcd_attn_feat_fwd_kernel<HEAD_DIM, NF, SHARE4>, shmem); \
            pcd_attn_feat_fwd_kernel<HEAD_DIM, NF, SHARE4><<<blocks, threads, shmem>>>( \
                M, C, K, coord_scale, rel_clamp, P_pcd, \
                reinterpret_cast<const float3*>(keypoints.data_ptr<float>()), \
                key_feat_ptr, \
                reinterpret_cast<const float3*>(points.data_ptr<float>()), \
                point_kfeat_ptr, point_vfeat_ptr, point_extras_ptr, \
                nn_idx.data_ptr<int>(), nn_count.data_ptr<int>(), \
                weights_ptr, bias_ptr, \
                out.data_ptr<float>(), \
                head_max.data_ptr<float>(), head_denom.data_ptr<float>()); \
        } while (0)

    if (proj_share_group == WARPS_PER_BLOCK) {
        switch (N_FEATURES) {
            case 0:
                LAUNCH_FEAT_KERNEL(0, true);
                break;
            case MODE_FULL:
                LAUNCH_FEAT_KERNEL(MODE_FULL, true);
                break;
            case MODE_ABS:
                LAUNCH_FEAT_KERNEL(MODE_ABS, true);
                break;
            case MODE_REL:
                LAUNCH_FEAT_KERNEL(MODE_REL, true);
                break;
            default:
                TORCH_CHECK(false, "Feature mode requires N_FEATURES in {96, 61, 39}");
        }
    } else {
        switch (N_FEATURES) {
            case 0:
                LAUNCH_FEAT_KERNEL(0, false);
                break;
            case MODE_FULL:
                LAUNCH_FEAT_KERNEL(MODE_FULL, false);
                break;
            case MODE_ABS:
                LAUNCH_FEAT_KERNEL(MODE_ABS, false);
                break;
            case MODE_REL:
                LAUNCH_FEAT_KERNEL(MODE_REL, false);
                break;
            default:
                TORCH_CHECK(false, "Feature mode requires N_FEATURES in {96, 61, 39}");
        }
    }

    #undef LAUNCH_FEAT_KERNEL
}

attn_out cuda_pcd_attn_fwd(
    torch::Tensor keypoints, torch::Tensor key_feat,
    torch::Tensor points, torch::Tensor point_kfeat, torch::Tensor point_vfeat, torch::Tensor point_extras,
    torch::Tensor nn_idx, torch::Tensor nn_count,
    torch::Tensor weights, torch::Tensor bias,
    int head_dim, int K, float coord_scale, float rel_clamp, int proj_share_group
){
    const int64_t M = keypoints.size(0);

    int C = 0;
    if (key_feat.defined() && key_feat.numel() > 0) C = key_feat.size(1);
    else if (point_vfeat.defined() && point_vfeat.numel() > 0) C = point_vfeat.size(1);
    else if (point_kfeat.defined() && point_kfeat.numel() > 0) C = point_kfeat.size(1);
    else if (weights.defined() && weights.numel() > 0) {
        const int share_factor = (proj_share_group == WARPS_PER_BLOCK) ? WARPS_PER_BLOCK : 1;
        C = weights.size(0) * weights.size(3) * share_factor;
    }

    if (C == 0) {
         return {torch::empty({M, 0}, keypoints.options()), nn_idx, nn_count, torch::empty({M, 0}, keypoints.options()), torch::empty({M, 0}, keypoints.options())};
    }

    // Detect feature mode based on weight shape
    int N_TOTAL = (weights.defined() && weights.numel() > 0) ? weights.size(2) : 0;
    bool is_feature_mode = (N_TOTAL == MODE_FULL || N_TOTAL == MODE_ABS || N_TOTAL == MODE_REL);

    if (is_feature_mode) {
        TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
        if (bias.defined() && bias.numel() > 0) {
            TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
        }
    }
    if (key_feat.defined() && key_feat.numel() > 0) TORCH_CHECK(key_feat.scalar_type() == torch::kFloat32, "key_feat must be float32");
    if (point_kfeat.defined() && point_kfeat.numel() > 0) TORCH_CHECK(point_kfeat.scalar_type() == torch::kFloat32, "point_kfeat must be float32");
    if (point_vfeat.defined() && point_vfeat.numel() > 0) TORCH_CHECK(point_vfeat.scalar_type() == torch::kFloat32, "point_vfeat must be float32");

    auto out = torch::empty({M, C}, keypoints.options().dtype(torch::kFloat32));

    const int H = C / head_dim;
    auto head_max = torch::empty({M, H}, torch::TensorOptions().device(points.device()).dtype(torch::kFloat32));
    auto head_denom = torch::empty({M, H}, torch::TensorOptions().device(points.device()).dtype(torch::kFloat32));

    const int mode_nf = is_feature_mode ? N_TOTAL : 0;
    switch (static_cast<int>(head_dim)) {
#ifdef PCD_ATTN_ONLY_HD96
        case 96: launch_attn_feat_fwd<96>(M, C, K, coord_scale, rel_clamp,
                    proj_share_group,
                    keypoints, key_feat, points, point_kfeat, point_vfeat, point_extras,
                    nn_idx, nn_count, weights, bias, out, head_max, head_denom, mode_nf); break;
        default: TORCH_CHECK(false, "This build only supports head_dim=96");
#else
        case 48: launch_attn_feat_fwd<48>(M, C, K, coord_scale, rel_clamp,
                    proj_share_group,
                    keypoints, key_feat, points, point_kfeat, point_vfeat, point_extras,
                    nn_idx, nn_count, weights, bias, out, head_max, head_denom, mode_nf); break;
        case 72: launch_attn_feat_fwd<72>(M, C, K, coord_scale, rel_clamp,
                    proj_share_group,
                    keypoints, key_feat, points, point_kfeat, point_vfeat, point_extras,
                    nn_idx, nn_count, weights, bias, out, head_max, head_denom, mode_nf); break;
        case 96: launch_attn_feat_fwd<96>(M, C, K, coord_scale, rel_clamp,
                    proj_share_group,
                    keypoints, key_feat, points, point_kfeat, point_vfeat, point_extras,
                    nn_idx, nn_count, weights, bias, out, head_max, head_denom, mode_nf); break;
        default: TORCH_CHECK(false, "Supported head_dim values are {48,72,96}");
#endif
    }

    return {out, nn_idx, nn_count, head_max, head_denom};
}


__device__ __forceinline__ void atomicAdd_custom(float* address, float val) {
    atomicAdd(address, val);
}


template <int HEAD_DIM, int N_FEATURES, bool SHARE_GROUP4>
__global__ void pcd_attn_bwd_kernel(
    const int M,
    const int C,
    const int K,
    const float coord_scale,
    const float rel_clamp,
    const int split_q,
    const int split_k,
    const int split_w,
    const int P_pcd,
    const int N_POINTS,
    const float3* __restrict__ key_pos,
    const float* __restrict__ key_feat,
    const float3* __restrict__ points,
    const float* __restrict__ point_kfeat,
    const float* __restrict__ point_vfeat,
    const float* __restrict__ point_extras,
    const int* __restrict__ nn_idx,
    const int* __restrict__ nn_count,
    const float* __restrict__ head_max,
    const float* __restrict__ head_denom,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_key_feat,
    float* __restrict__ grad_point_kfeat,
    float* __restrict__ grad_point_vfeat,
    float* __restrict__ grad_weights,
    float* __restrict__ grad_bias,
    const float* __restrict__ out
) {
    // Feature modes only (96, 61, 39)
    constexpr bool IS_FULL = (N_FEATURES == MODE_FULL);
    constexpr bool IS_ABS  = (N_FEATURES == MODE_ABS);
    constexpr bool IS_REL  = (N_FEATURES == MODE_REL);
    constexpr bool IS_FEATURE_MODE = IS_FULL || IS_ABS || IS_REL;
    constexpr int LANES = (HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE;
    constexpr bool ENABLE_BWD_PVF = (PCD_BWD_DISABLE_PVF == 0);
    constexpr bool ENABLE_BWD_PKF = (PCD_BWD_DISABLE_PKF == 0);
    constexpr bool ENABLE_BWD_GRADW = (PCD_BWD_DISABLE_GRADW == 0);
    constexpr bool ENABLE_BWD_BIAS = (PCD_BWD_DISABLE_BIAS == 0);
    constexpr bool ENABLE_BWD_PROJ_QK = (PCD_BWD_DISABLE_PROJ_QK == 0);
    constexpr bool ENABLE_BWD_PROJ_V = (PCD_BWD_DISABLE_PROJ_V == 0);
    constexpr bool ENABLE_BWD_GW_Q = (PCD_BWD_DISABLE_GW_Q == 0);
    constexpr bool ENABLE_BWD_GW_K = (PCD_BWD_DISABLE_GW_K == 0);
    constexpr bool ENABLE_BWD_GW_V = (PCD_BWD_DISABLE_GW_V == 0);
    using feat_t = smem_feat_t;
    using proj_t = smem_proj_t;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int key_idx = blockIdx.x;
    if (key_idx >= M) return;
    constexpr int PROJ_SLOTS = SHARE_GROUP4 ? 1 : WARPS_PER_BLOCK;
    const int H = C / HEAD_DIM;
    const int H_proj = SHARE_GROUP4 ? (H / WARPS_PER_BLOCK) : H;

    const int q_bin = key_idx % split_q;
    const int k_bin = key_idx % split_k;
    const int w_bin = key_idx % split_w;
    const size_t q_offset   = grad_key_feat ? static_cast<size_t>(q_bin) * M * C : 0;
    const size_t pkf_offset = grad_point_kfeat ? static_cast<size_t>(k_bin) * N_POINTS * C : 0;
    const size_t pvf_offset = grad_point_vfeat ? static_cast<size_t>(k_bin) * N_POINTS * C : 0;
    const size_t w_offset   = grad_weights ? static_cast<size_t>(w_bin) * H_proj * 3 * N_FEATURES * HEAD_DIM : 0;
    const size_t b_offset   = grad_bias ? static_cast<size_t>(w_bin) * H * 3 * HEAD_DIM : 0;

    // Shared Memory - Reuse layout from FWD
    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned char* smem = smem_raw;

    constexpr int KTILE = BWD_KTILE;

    feat_t* sh_q_feat = nullptr;  // For feature mode
    feat_t* sh_k_feat = nullptr;  // For feature mode (per warp)
    int* sh_pid = nullptr;
    proj_t* sh_bq_proj = nullptr;
    proj_t* sh_bk_proj = nullptr;
    proj_t* sh_bq_qproj = nullptr;
    float* sh_wattn = nullptr;
    float* sh_dlogit = nullptr;

    if constexpr (IS_FEATURE_MODE) {
        // Feature mode: Q features + per-warp PIDs + per-warp K features tile
        sh_q_feat = reinterpret_cast<feat_t*>(smem);
        smem += align16(N_FEATURES * sizeof(feat_t));
        sh_pid = reinterpret_cast<int*>(smem);
        smem += align16(KTILE * sizeof(int));
        sh_k_feat = reinterpret_cast<feat_t*>(smem);
        smem += align16(static_cast<size_t>(KTILE) * N_FEATURES * sizeof(feat_t));
        sh_bq_proj = reinterpret_cast<proj_t*>(smem);
        smem += align16(static_cast<size_t>(PROJ_SLOTS) * KTILE * HEAD_DIM * sizeof(proj_t));
        sh_bk_proj = reinterpret_cast<proj_t*>(smem);
        smem += align16(static_cast<size_t>(PROJ_SLOTS) * KTILE * HEAD_DIM * sizeof(proj_t));

        if constexpr (ENABLE_BWD_GRADW) {
            sh_wattn = reinterpret_cast<float*>(smem);
            smem += align16(static_cast<size_t>(WARPS_PER_BLOCK) * KTILE * sizeof(float));
            sh_dlogit = reinterpret_cast<float*>(smem);
            smem += align16(static_cast<size_t>(WARPS_PER_BLOCK) * KTILE * sizeof(float));
        }

        // Add qproj to smem initialization in bwd kernel
        sh_bq_qproj = reinterpret_cast<proj_t*>(smem);
        smem += align16(static_cast<size_t>(PROJ_SLOTS) * HEAD_DIM * sizeof(proj_t));

    } else {
        // Plain mode: just PIDs
        sh_pid = reinterpret_cast<int*>(smem);
        smem += align16(KTILE * sizeof(int));
    }

    const int count = nn_count[key_idx];
    if (count == 0) return;

    const float3 qpos = key_pos[key_idx];
    // key_pos/points are normalized to [0, 1]; convert back to "original" coord space [-1, 1]
    const float3 qpos_o = make_float3(qpos.x * 2.0f - 1.0f, qpos.y * 2.0f - 1.0f, qpos.z * 2.0f - 1.0f);

    // Feature mode: compute Q features (abs only; rel/extras handled separately)
    if constexpr (IS_FEATURE_MODE) {
        for (int f = threadIdx.x; f < N_FEATURES; f += blockDim.x) {
            float val = 0.0f;
            if constexpr (IS_FULL) {
                if (f < NUM_ABS) {
                    val = compute_abs_feature(f, qpos_o);
                }
            } else if constexpr (IS_ABS) {
                if (f < NUM_ABS) {
                    val = compute_abs_feature(f, qpos_o);
                }
            } else {  // IS_REL
                val = 0.0f;
            }
            sh_q_feat[f] = val;
        }
        __syncthreads();
    }

    const float inv_sqrt_d = rsqrtf(static_cast<float>(HEAD_DIM));

    const float* qbase = (key_feat) ? key_feat + static_cast<size_t>(key_idx) * C : nullptr;

    // One warp per head (4 heads per block).
    for (int h0 = 0; h0 < H; h0 += WARPS_PER_BLOCK) {
        const int h = h0 + warp_id;
        const int h_proj = SHARE_GROUP4 ? (h0 / WARPS_PER_BLOCK) : h;
        const bool active = (h < H);

        const float* grad_out_f = (active && grad_out) ? (grad_out + static_cast<size_t>(key_idx) * C + h * HEAD_DIM) : nullptr;
        const float* out_f      = (active && out)      ? (out      + static_cast<size_t>(key_idx) * C + h * HEAD_DIM) : nullptr;
        const float* qf           = (active && qbase)    ? (qbase    + h * HEAD_DIM) : nullptr;
        const float* b_head        = (active && bias)     ? (bias     + h * 3 * HEAD_DIM) : nullptr;

        const float m_logit  = active ? head_max[static_cast<size_t>(key_idx) * H + h] : 0.0f;
        const float denom    = active ? head_denom[static_cast<size_t>(key_idx) * H + h] : 1.0f;
        const float inv_denom = 1.0f / denom;

        float gd[LANES];

        // Compute grad_dot_out = dot(grad_out, out) inline without storing od[]
        float grad_dot_out = 0.f;
#pragma unroll
        for (int li = 0; li < LANES; ++li) {
            const int dd = lane + li * WARP_SIZE;
            gd[li] = (grad_out_f && dd < HEAD_DIM) ? grad_out_f[dd] : 0.f;
            float ov = (out_f && dd < HEAD_DIM) ? out_f[dd] : 0.f;
            grad_dot_out += gd[li] * ov;
        }
        grad_dot_out = warp_reduce_sum(grad_dot_out);
        grad_dot_out = __shfl_sync(FULL_MASK, grad_dot_out, 0);

        float dL_dq[LANES] = {0.f};

        if constexpr (IS_FEATURE_MODE) {
            const float* w_base = ((h_proj < H_proj) && weights) ? weights + h_proj * 3 * N_FEATURES * HEAD_DIM : nullptr;
            const float* w_q = w_base;
            const float* w_k = w_base ? (w_base + N_FEATURES * HEAD_DIM) : nullptr;
            const float* w_v = w_base ? (w_base + 2 * N_FEATURES * HEAD_DIM) : nullptr;

            float* sh_wattn_w = ENABLE_BWD_GRADW ? (sh_wattn + static_cast<size_t>(warp_id) * KTILE) : nullptr;
            float* sh_dlogit_w = ENABLE_BWD_GRADW ? (sh_dlogit + static_cast<size_t>(warp_id) * KTILE) : nullptr;
            float gb_k_accum[LANES] = {0.f};
            float sum_w_attn_accum = 0.f;


            for (int k0 = 0; k0 < count; k0 += KTILE) {
                const int kt = (k0 + KTILE <= count) ? KTILE : (count - k0);

                int* sh_pid_w = sh_pid;
                feat_t* sh_k_feat_w = sh_k_feat;

                for (int t = tid; t < kt; t += blockDim.x) {
                    int pid = nn_idx[static_cast<size_t>(key_idx) * K + (k0 + t)];
                    if (pid < 0 || pid >= N_POINTS) pid = 0;
                    sh_pid_w[t] = pid;
                }
                __syncthreads();

                for (int t = tid; t < kt; t += blockDim.x) {
                    const int pid = sh_pid_w[t];
                    const float3 pp = points[pid];
                    const float3 pp_o = make_float3(pp.x * 2.0f - 1.0f, pp.y * 2.0f - 1.0f, pp.z * 2.0f - 1.0f);
                    const float* extras = (point_extras && P_pcd > 0) ? point_extras + static_cast<size_t>(pid) * P_pcd : nullptr;
                    encode_neighbor_features<KTILE, IS_FULL, IS_ABS, IS_REL>(
                        sh_k_feat_w, t, pp_o, qpos_o, coord_scale, rel_clamp, extras, P_pcd);
                }
                __syncthreads();

                fill_fourier_features<KTILE, IS_FULL, IS_ABS, IS_REL>(
                    sh_k_feat_w, tid, blockDim.x, kt);
                __syncthreads();

                const int proj_warp = SHARE_GROUP4 ? 0 : warp_id;
                proj_t* bq_proj_w = sh_bq_proj + static_cast<size_t>(proj_warp) * KTILE * HEAD_DIM;
                proj_t* bk_proj_w = sh_bk_proj + static_cast<size_t>(proj_warp) * KTILE * HEAD_DIM;
                const bool has_proj_qk = ENABLE_BWD_PROJ_QK && (w_q || w_k);
                const bool use_proj_q = ENABLE_BWD_PROJ_QK && (w_q != nullptr);
                const bool use_proj_k = ENABLE_BWD_PROJ_QK && (w_k != nullptr);
                const bool use_proj_v = ENABLE_BWD_PROJ_V && (w_v != nullptr);

                proj_t* qp_q = sh_bq_qproj + static_cast<size_t>(proj_warp) * HEAD_DIM;

                if (has_proj_qk) {
                    if constexpr (SHARE_GROUP4) {
                        if (k0 == 0) {
                            if (warp_id == 0) {
                                const float* w_ptrs_q[BIAS_COUNT] = {w_q, nullptr, nullptr};
                                proj_t* q_proj_ptrs_q[BIAS_COUNT] = {qp_q, nullptr, nullptr};
                                compute_q_bias_proj<HEAD_DIM, N_FEATURES, IS_FULL, IS_ABS, IS_REL>(
                                    lane, w_ptrs_q, q_proj_ptrs_q, sh_q_feat);
                            }
                        }
                        __syncthreads();

                        fma_bias_proj_tile_single_split_tk<HEAD_DIM, N_FEATURES, IS_FULL, IS_ABS, IS_REL, BIAS_Q, KTILE>(
                            lane, warp_id, WARPS_PER_BLOCK, kt, w_q, bq_proj_w, qp_q, sh_k_feat_w);
                        fma_bias_proj_tile_single_split_tk<HEAD_DIM, N_FEATURES, IS_FULL, IS_ABS, IS_REL, BIAS_K, KTILE>(
                            lane, warp_id, WARPS_PER_BLOCK, kt, w_k, bk_proj_w, nullptr, sh_k_feat_w);
                    } else {
                        const float* w_ptrs[BIAS_COUNT] = {w_q, w_k, w_v};
                        proj_t* proj_ptrs[BIAS_COUNT] = {bq_proj_w, bk_proj_w, nullptr};
                        const proj_t* q_proj_ptrs[BIAS_COUNT] = {qp_q, nullptr, nullptr};
                        if (k0 == 0) {
                            proj_t* n_q_proj_ptrs[BIAS_COUNT] = {qp_q, nullptr, nullptr};
                            compute_q_bias_proj<HEAD_DIM, N_FEATURES, IS_FULL, IS_ABS, IS_REL>(
                                lane, w_ptrs, n_q_proj_ptrs, sh_q_feat);
                        }

                        fma_bias_proj_tile_unified<HEAD_DIM, N_FEATURES, IS_FULL, IS_ABS, IS_REL, KTILE>(
                            lane, kt, w_ptrs, proj_ptrs, q_proj_ptrs, sh_k_feat_w);
                    }
                }
                if (has_proj_qk && SHARE_GROUP4) __syncthreads();
                else if (has_proj_qk) __syncwarp();

                if (active) {
                    const bool need_wb_staging = ENABLE_BWD_GRADW && (grad_weights && weights);
                    const bool need_bias_accum = ENABLE_BWD_BIAS && grad_bias;
                    for (int tk = 0; tk < KTILE; ++tk) {
                        if (tk >= kt) continue;
                        const int pid = sh_pid_w[tk];
                        const float* kf_ptr = point_kfeat ? point_kfeat + (static_cast<size_t>(pid) * C + h * HEAD_DIM) : nullptr;
                        const float* vf_ptr = point_vfeat ? point_vfeat + (static_cast<size_t>(pid) * C + h * HEAD_DIM) : nullptr;

                        float q_eff[LANES];
                        float k_eff[LANES];
                        float dot = 0.f;
                        float vg_sum = 0.f;

#pragma unroll
                        for (int li = 0; li < LANES; ++li) {
                            const int dd = lane + li * WARP_SIZE;
                            if (dd >= HEAD_DIM) { q_eff[li] = 0.f; k_eff[li] = 0.f; continue; }

                            float qe = qf ? qf[dd] : 0.f;
                            if (b_head) qe += b_head[dd];
                            if (use_proj_q) qe += half_to_float(bq_proj_w[tk * HEAD_DIM + dd]);

                            float ke = kf_ptr ? kf_ptr[dd] : 0.f;
                            if (b_head) ke += b_head[HEAD_DIM + dd];
                            if (use_proj_k) ke += half_to_float(bk_proj_w[tk * HEAD_DIM + dd]);

                            q_eff[li] = qe;
                            k_eff[li] = ke;
                            dot += qe * ke;

                            float ve = vf_ptr ? vf_ptr[dd] : 0.f;
                            if (b_head) ve += b_head[2 * HEAD_DIM + dd];
                            if (use_proj_v) {
                                ve += compute_point_bias_proj_dim<HEAD_DIM, N_FEATURES, KTILE>(
                                    dd, tk, w_v, sh_k_feat_w);
                            }
                            vg_sum += ve * gd[li];
                        }
                        dot = warp_reduce_sum(dot);
                        dot = __shfl_sync(FULL_MASK, dot, 0);
                        const float logit = dot * inv_sqrt_d;

                        const float w_attn = expf(logit - m_logit) * inv_denom;
                        if (need_bias_accum) sum_w_attn_accum += w_attn;

                        if (ENABLE_BWD_PVF && grad_point_vfeat) {
#pragma unroll
                            for (int li = 0; li < LANES; ++li) {
                                const int dd = lane + li * WARP_SIZE;
                                if (dd < HEAD_DIM) {
                                    atomicAdd_custom(&grad_point_vfeat[pvf_offset + static_cast<size_t>(pid) * C + h * HEAD_DIM + dd],
                                        w_attn * gd[li]);
                                }
                            }
                        }

                        vg_sum = warp_reduce_sum(vg_sum);
                        vg_sum = __shfl_sync(FULL_MASK, vg_sum, 0);
                        const float dL_dlogit = w_attn * (vg_sum - grad_dot_out) * inv_sqrt_d;

                        if (need_wb_staging) {
                            if (lane == 0) {
                                sh_wattn_w[tk] = w_attn;
                                sh_dlogit_w[tk] = dL_dlogit;
                            }
                        }

                        if (ENABLE_BWD_PKF && grad_point_kfeat) {
                            for (int li = 0; li < LANES; ++li) {
                                const int dd = lane + li * WARP_SIZE;
                                if (dd < HEAD_DIM) {
                                    atomicAdd_custom(&grad_point_kfeat[pkf_offset + static_cast<size_t>(pid) * C + h * HEAD_DIM + dd],
                                        dL_dlogit * q_eff[li]);
                                }
                            }
                        }
#pragma unroll
                        for (int li = 0; li < LANES; ++li) {
                            if ((lane + li * WARP_SIZE) < HEAD_DIM) {
                                dL_dq[li] += dL_dlogit * k_eff[li];
                                if (need_bias_accum) gb_k_accum[li] += dL_dlogit * q_eff[li];
                            }
                        }
                    }
                    if (need_wb_staging) __syncwarp();

                    if (ENABLE_BWD_GRADW && grad_weights && weights) {
                        float* gw_base = grad_weights + w_offset + h_proj * 3 * N_FEATURES * HEAD_DIM;
                        // Smaller tile reduces per-thread temporary arrays (register pressure).
                        constexpr int FTILE = F_TILE;
                        for (int li = 0; li < LANES; ++li) {
                            const int dd = lane + li * WARP_SIZE;
                            if (dd >= HEAD_DIM) continue;

                            // Reuse per-neighbor scalar terms across all feature tiles.
                            float dQ_t[KTILE];
                            float dK_t[KTILE];
                            float sum_dL_dQ = 0.f;
                            for (int tk = 0; tk < KTILE; ++tk) {
                                if (tk >= kt) continue;
                                const float dlogit = sh_dlogit_w[tk];
                                const int pid = sh_pid_w[tk];
                                const float* kf_ptr = point_kfeat ? point_kfeat + (static_cast<size_t>(pid) * C + h * HEAD_DIM) : nullptr;
                                if constexpr (ENABLE_BWD_GW_Q) {
                                    float k_t = kf_ptr ? kf_ptr[dd] : 0.f;
                                    if (b_head) k_t += b_head[HEAD_DIM + dd];
                                    if (use_proj_k) k_t += half_to_float(bk_proj_w[tk * HEAD_DIM + dd]);
                                    const float dQ = dlogit * k_t;
                                    dQ_t[tk] = dQ;
                                    sum_dL_dQ += dQ;
                                }
                                if constexpr (ENABLE_BWD_GW_K) {
                                    float q_t = qf ? qf[dd] : 0.f;
                                    if (b_head) q_t += b_head[dd];
                                    if (use_proj_q) q_t += half_to_float(bq_proj_w[tk * HEAD_DIM + dd]);
                                    dK_t[tk] = dlogit * q_t;
                                }
                            }

                            for (int f0 = 0; f0 < N_FEATURES; f0 += FTILE) {
                                const int f_max = min(FTILE, N_FEATURES - f0);
                                float gw_q[FTILE] = {0.f};
                                float gw_k[FTILE] = {0.f};
                                float gw_v[FTILE] = {0.f};

                                for (int tk = 0; tk < KTILE; ++tk) {
                                    if (tk >= kt) continue;
                                    const float wattn = sh_wattn_w[tk];
                                    const float dQ = ENABLE_BWD_GW_Q ? dQ_t[tk] : 0.f;
                                    const float dK = ENABLE_BWD_GW_K ? dK_t[tk] : 0.f;
                                    const float dV = ENABLE_BWD_GW_V ? (wattn * gd[li]) : 0.f;

                                    for (int fi = 0; fi < f_max; ++fi) {
                                        const int f = f0 + fi;
                                        const float val_k = sh_k_feat_w[static_cast<size_t>(f) * KTILE + tk];
                                        if constexpr (ENABLE_BWD_GW_K) {
                                            gw_k[fi] += dK * val_k;
                                        }
                                        if constexpr (ENABLE_BWD_GW_V) {
                                            gw_v[fi] += dV * val_k;
                                        }
                                        if constexpr (ENABLE_BWD_GW_Q) {
                                            if constexpr (IS_FULL) {
                                                if (f >= NUM_ABS && f < (NUM_ABS + NUM_REL))
                                                    gw_q[fi] += dQ * val_k;
                                            } else if constexpr (IS_REL) {
                                                if (f < NUM_REL)
                                                    gw_q[fi] += dQ * val_k;
                                            }
                                        }
                                    }
                                }

                                if constexpr (ENABLE_BWD_GW_Q) {
                                    for (int fi = 0; fi < f_max; ++fi) {
                                        const int f = f0 + fi;
                                        if constexpr (IS_ABS) {
                                            gw_q[fi] += sum_dL_dQ * sh_q_feat[f];
                                        } else if constexpr (IS_FULL) {
                                            if (f < NUM_ABS || f >= (NUM_ABS + NUM_REL))
                                                gw_q[fi] += sum_dL_dQ * sh_q_feat[f];
                                        } else if constexpr (IS_REL) {
                                            if (f >= NUM_REL)
                                                gw_q[fi] += sum_dL_dQ * sh_q_feat[f];
                                        }
                                    }
                                }

                                for (int fi = 0; fi < f_max; ++fi) {
                                    const int f = f0 + fi;
                                    if constexpr (ENABLE_BWD_GW_Q) {
                                        atomicAdd_custom(gw_base + f * HEAD_DIM + dd, gw_q[fi]);
                                    }
                                    if constexpr (ENABLE_BWD_GW_K) {
                                        atomicAdd_custom(gw_base + N_FEATURES * HEAD_DIM + f * HEAD_DIM + dd, gw_k[fi]);
                                    }
                                    if constexpr (ENABLE_BWD_GW_V) {
                                        atomicAdd_custom(gw_base + 2 * N_FEATURES * HEAD_DIM + f * HEAD_DIM + dd, gw_v[fi]);
                                    }
                                }
                            }
                        }
                    }

                }
                __syncwarp();
            }
            if (ENABLE_BWD_BIAS && active && grad_bias) {
                float* gb_base = grad_bias + b_offset + h * 3 * HEAD_DIM;
#pragma unroll
                for (int li = 0; li < LANES; ++li) {
                    const int dd = lane + li * WARP_SIZE;
                    if (dd < HEAD_DIM) {
                        atomicAdd_custom(gb_base + dd, dL_dq[li]);
                        atomicAdd_custom(gb_base + HEAD_DIM + dd, gb_k_accum[li]);
                        atomicAdd_custom(gb_base + 2 * HEAD_DIM + dd, sum_w_attn_accum * gd[li]);
                    }
                }
            }

        } else {
            // Plain backward: no positional weights/bias.
            float qd[LANES];
#pragma unroll
            for (int li = 0; li < LANES; ++li) {
                const int dd = lane + li * WARP_SIZE;
                qd[li] = (qf && dd < HEAD_DIM) ? qf[dd] : 0.f;
            }

            if (active) {
                for (int k = 0; k < count; ++k) {
                    int pid = nn_idx[static_cast<size_t>(key_idx) * K + k];
                    if (pid < 0 || pid >= N_POINTS) pid = 0;

                    const float* kf_ptr = point_kfeat ? point_kfeat + (static_cast<size_t>(pid) * C + h * HEAD_DIM) : nullptr;
                    const float* vf_ptr = point_vfeat ? point_vfeat + (static_cast<size_t>(pid) * C + h * HEAD_DIM) : nullptr;

                    float dot = 0.f;
                    float vg_sum = 0.f;
                    float kd[LANES];
#pragma unroll
                    for (int li = 0; li < LANES; ++li) {
                        const int dd = lane + li * WARP_SIZE;
                        kd[li] = (kf_ptr && dd < HEAD_DIM) ? kf_ptr[dd] : 0.f;
                        float vd = (vf_ptr && dd < HEAD_DIM) ? vf_ptr[dd] : 0.f;
                        if (dd < HEAD_DIM) dot += qd[li] * kd[li];
                        vg_sum += vd * gd[li];
                    }
                    dot = warp_reduce_sum(dot);
                    dot = __shfl_sync(FULL_MASK, dot, 0);
                    const float logit = dot * inv_sqrt_d;

                    const float w_attn = expf(logit - m_logit) * inv_denom;
                    if (grad_point_vfeat) {
#pragma unroll
                        for (int li = 0; li < LANES; ++li) {
                            const int dd = lane + li * WARP_SIZE;
                            if (dd < HEAD_DIM) {
                                atomicAdd_custom(&grad_point_vfeat[pvf_offset + static_cast<size_t>(pid) * C + h * HEAD_DIM + dd],
                                    w_attn * gd[li]);
                            }
                        }
                    }

                    vg_sum = warp_reduce_sum(vg_sum);
                    vg_sum = __shfl_sync(FULL_MASK, vg_sum, 0);
                    const float dL_dlogit = w_attn * (vg_sum - grad_dot_out) * inv_sqrt_d;

                    if (grad_point_kfeat) {
#pragma unroll
                        for (int li = 0; li < LANES; ++li) {
                            const int dd = lane + li * WARP_SIZE;
                            if (dd < HEAD_DIM) {
                                atomicAdd_custom(&grad_point_kfeat[pkf_offset + static_cast<size_t>(pid) * C + h * HEAD_DIM + dd],
                                    dL_dlogit * qd[li]);
                            }
                        }
                    }

#pragma unroll
                    for (int li = 0; li < LANES; ++li) {
                        if ((lane + li * WARP_SIZE) < HEAD_DIM) dL_dq[li] += dL_dlogit * kd[li];
                    }
                }
            }
        }

        if (active && grad_key_feat) {
#pragma unroll
            for (int li = 0; li < LANES; ++li) {
                const int dd = lane + li * WARP_SIZE;
                if (dd < HEAD_DIM) {
                    grad_key_feat[q_offset + static_cast<size_t>(key_idx) * C + h * HEAD_DIM + dd] = dL_dq[li];
                }
            }
        }

        __syncthreads();
    }
}

template <int HEAD_DIM>
static void launch_attn_bwd(
    int M, int C, int K, float coord_scale, float rel_clamp, int split_q, int split_k, int proj_share_group,
    torch::Tensor grad_out,
    torch::Tensor key_feat,
    torch::Tensor keypoints, torch::Tensor points,
    torch::Tensor point_kfeat, torch::Tensor point_vfeat, torch::Tensor point_extras,
    torch::Tensor nn_idx, torch::Tensor nn_count,
    torch::Tensor head_max, torch::Tensor head_denom,
    torch::Tensor weights, torch::Tensor bias,
    torch::Tensor out,
    torch::Tensor grad_key_feat,
    torch::Tensor grad_point_kfeat,
    torch::Tensor grad_point_vfeat,
    torch::Tensor grad_weights,
    torch::Tensor grad_bias
){
    const int threads = WARPS_PER_BLOCK * 32;
    const int blocks = M;

    int N_TOTAL = (weights.defined() && weights.numel() > 0) ? weights.size(2) : 0;
    int P_pcd = (point_extras.defined() && point_extras.numel() > 0) ? point_extras.size(1) : 0;
    const int N_POINTS = points.size(0);
    const int split_w = (split_q > split_k) ? split_q : split_k;
    const int proj_slots = (proj_share_group == WARPS_PER_BLOCK) ? 1 : WARPS_PER_BLOCK;

    const int N_FEATURES = (N_TOTAL == MODE_FULL) ? MODE_FULL :
                           (N_TOTAL == MODE_ABS) ? MODE_ABS :
                           (N_TOTAL == MODE_REL) ? MODE_REL : 0;
    const bool is_feature_mode = (N_FEATURES != 0);

    constexpr int KTILE = BWD_KTILE;
    size_t shmem = 0;
    if (is_feature_mode) {
        const size_t feat_sz = smem_feat_sizeof();
        const size_t proj_sz = smem_proj_sizeof();
        // Feature mode shared memory: Q features + PIDs + K features
        size_t sz_q_feat = align16(N_TOTAL * feat_sz);  // Q features (96/61/39)
        size_t sz_pid    = align16(KTILE * sizeof(int));
        size_t sz_k_feat = align16(static_cast<size_t>(KTILE) * N_TOTAL * feat_sz);  // K features tile shared across warps
        shmem = sz_q_feat + sz_pid + sz_k_feat;
        // Bias projection buffers
        shmem += align16(static_cast<size_t>(proj_slots) * KTILE * HEAD_DIM * proj_sz) * 2; // bq/bk
        shmem += align16(static_cast<size_t>(proj_slots) * HEAD_DIM * proj_sz); // bq_q
#if PCD_BWD_DISABLE_GRADW == 0
        shmem += align16(static_cast<size_t>(WARPS_PER_BLOCK) * KTILE * sizeof(float)) * 2; // w_attn/dL_dlogit
#endif
    } else {
        shmem = align16(KTILE * sizeof(int)); // plain mode needs PID
    }

    const float* key_feat_ptr = (key_feat.defined() && key_feat.numel() > 0) ? key_feat.data_ptr<float>() : nullptr;
    const float* point_kfeat_ptr = (point_kfeat.defined() && point_kfeat.numel() > 0) ? point_kfeat.data_ptr<float>() : nullptr;
    const float* point_vfeat_ptr = (point_vfeat.defined() && point_vfeat.numel() > 0) ? point_vfeat.data_ptr<float>() : nullptr;
    const float* point_extras_ptr = (point_extras.defined() && point_extras.numel() > 0) ? point_extras.data_ptr<float>() : nullptr;

    const float* weights_ptr = (weights.defined() && weights.numel() > 0) ? weights.data_ptr<float>() : nullptr;
    const float* bias_ptr = (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;

    float* gq_ptr = (grad_key_feat.defined()) ? grad_key_feat.data_ptr<float>() : nullptr;
    float* gk_ptr = (grad_point_kfeat.defined()) ? grad_point_kfeat.data_ptr<float>() : nullptr;
    float* gv_ptr = (grad_point_vfeat.defined()) ? grad_point_vfeat.data_ptr<float>() : nullptr;
    float* gw_ptr = (grad_weights.defined()) ? grad_weights.data_ptr<float>() : nullptr;
    float* gb_ptr = (grad_bias.defined()) ? grad_bias.data_ptr<float>() : nullptr;

    #define LAUNCH_BWD_KERNEL(NFEAT, SHARE4) \
        do { \
        maybe_set_smem(pcd_attn_bwd_kernel<HEAD_DIM, NFEAT, SHARE4>, shmem); \
        pcd_attn_bwd_kernel<HEAD_DIM, NFEAT, SHARE4><<<blocks, threads, shmem>>>( \
            M, C, K, coord_scale, rel_clamp, split_q, split_k, split_w, P_pcd, N_POINTS, \
            reinterpret_cast<float3*>(keypoints.data_ptr<float>()), \
            key_feat_ptr, \
            reinterpret_cast<float3*>(points.data_ptr<float>()), \
            point_kfeat_ptr, point_vfeat_ptr, point_extras_ptr, \
            nn_idx.data_ptr<int>(), nn_count.data_ptr<int>(), \
            head_max.data_ptr<float>(), head_denom.data_ptr<float>(), \
            weights_ptr, bias_ptr, \
            grad_out.data_ptr<float>(), \
            gq_ptr, gk_ptr, gv_ptr, gw_ptr, gb_ptr, \
            out.data_ptr<float>() \
        ); \
        } while (0)

    if (proj_share_group == WARPS_PER_BLOCK) {
        switch(N_FEATURES) {
            case MODE_FULL: LAUNCH_BWD_KERNEL(MODE_FULL, true); break;
            case MODE_ABS:  LAUNCH_BWD_KERNEL(MODE_ABS, true); break;
            case MODE_REL:  LAUNCH_BWD_KERNEL(MODE_REL, true); break;
            default: LAUNCH_BWD_KERNEL(0, true); break;
        }
    } else {
        switch(N_FEATURES) {
            case MODE_FULL: LAUNCH_BWD_KERNEL(MODE_FULL, false); break;
            case MODE_ABS:  LAUNCH_BWD_KERNEL(MODE_ABS, false); break;
            case MODE_REL:  LAUNCH_BWD_KERNEL(MODE_REL, false); break;
            default: LAUNCH_BWD_KERNEL(0, false); break;
        }
    }

    #undef LAUNCH_BWD_KERNEL
}


std::vector<torch::Tensor> cuda_pcd_attn_bwd(
    torch::Tensor grad_out,
    torch::Tensor key_feat,
    torch::Tensor keypoints,
    torch::Tensor points,
    torch::Tensor point_kfeat,
    torch::Tensor point_vfeat,
    torch::Tensor point_extras,
    torch::Tensor nn_idx,
    torch::Tensor nn_count,
    torch::Tensor head_max,
    torch::Tensor head_denom,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor out,
    int head_dim,
    float coord_scale,
    float rel_clamp,
    int split_q,
    int split_k,
    int proj_share_group
) {
    const int M = keypoints.size(0);
    const int C = grad_out.size(1);
    const int K = nn_idx.size(1);
    const int split_w = (split_q > split_k) ? split_q : split_k;

    torch::Tensor grad_key_feat;
    if (key_feat.defined() && key_feat.numel() > 0) {
        if (split_q > 1) {
            grad_key_feat = torch::zeros(
                {split_q, key_feat.size(0), key_feat.size(1)},
                key_feat.options().dtype(torch::kFloat32)
            );
        } else {
            grad_key_feat = torch::zeros_like(key_feat, key_feat.options().dtype(torch::kFloat32));
        }
    } else {
        grad_key_feat = torch::zeros_like(key_feat, key_feat.options().dtype(torch::kFloat32));
    }

    torch::Tensor grad_point_kfeat;
    if (point_kfeat.defined() && point_kfeat.numel() > 0) {
        grad_point_kfeat = torch::zeros({split_k, point_kfeat.size(0), point_kfeat.size(1)}, point_kfeat.options().dtype(torch::kFloat32));
    } else {
        grad_point_kfeat = torch::zeros({split_k, 0}, point_kfeat.options().dtype(torch::kFloat32));
    }

    torch::Tensor grad_point_vfeat;
    if (point_vfeat.defined() && point_vfeat.numel() > 0) {
        grad_point_vfeat = torch::zeros({split_k, point_vfeat.size(0), point_vfeat.size(1)}, point_vfeat.options().dtype(torch::kFloat32));
    } else {
        grad_point_vfeat = torch::zeros({split_k, 0}, point_vfeat.options().dtype(torch::kFloat32));
    }

    torch::Tensor grad_weights;
    if (weights.defined() && weights.numel() > 0 && weights.requires_grad()) {
        auto sizes = weights.sizes().vec();
        sizes.insert(sizes.begin(), split_w);
        grad_weights = torch::zeros(sizes, weights.options().dtype(torch::kFloat32));
    }
    torch::Tensor grad_bias;
    if (bias.defined() && bias.numel() > 0 && bias.requires_grad()) {
        auto sizes = bias.sizes().vec();
        sizes.insert(sizes.begin(), split_w);
        grad_bias = torch::zeros(sizes, bias.options().dtype(torch::kFloat32));
    }

    TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32, "grad_out must be float32");
    if (key_feat.defined() && key_feat.numel() > 0) TORCH_CHECK(key_feat.scalar_type() == torch::kFloat32, "key_feat must be float32");
    if (point_kfeat.defined() && point_kfeat.numel() > 0) TORCH_CHECK(point_kfeat.scalar_type() == torch::kFloat32, "point_kfeat must be float32");
    if (point_vfeat.defined() && point_vfeat.numel() > 0) TORCH_CHECK(point_vfeat.scalar_type() == torch::kFloat32, "point_vfeat must be float32");

    switch (static_cast<int>(head_dim)) {
#ifdef PCD_ATTN_ONLY_HD96
        case 96: launch_attn_bwd<96>(M, C, K, coord_scale, rel_clamp, split_q, split_k, proj_share_group, grad_out, key_feat, keypoints, points, point_kfeat, point_vfeat, point_extras, nn_idx, nn_count, head_max, head_denom, weights, bias, out, grad_key_feat, grad_point_kfeat, grad_point_vfeat, grad_weights, grad_bias); break;
        default: TORCH_CHECK(false, "This build only supports head_dim=96");
#else
        case 48: launch_attn_bwd<48>(M, C, K, coord_scale, rel_clamp, split_q, split_k, proj_share_group, grad_out, key_feat, keypoints, points, point_kfeat, point_vfeat, point_extras, nn_idx, nn_count, head_max, head_denom, weights, bias, out, grad_key_feat, grad_point_kfeat, grad_point_vfeat, grad_weights, grad_bias); break;
        case 72: launch_attn_bwd<72>(M, C, K, coord_scale, rel_clamp, split_q, split_k, proj_share_group, grad_out, key_feat, keypoints, points, point_kfeat, point_vfeat, point_extras, nn_idx, nn_count, head_max, head_denom, weights, bias, out, grad_key_feat, grad_point_kfeat, grad_point_vfeat, grad_weights, grad_bias); break;
        case 96: launch_attn_bwd<96>(M, C, K, coord_scale, rel_clamp, split_q, split_k, proj_share_group, grad_out, key_feat, keypoints, points, point_kfeat, point_vfeat, point_extras, nn_idx, nn_count, head_max, head_denom, weights, bias, out, grad_key_feat, grad_point_kfeat, grad_point_vfeat, grad_weights, grad_bias); break;
        default: TORCH_CHECK(false, "Supported head_dim values are {48,72,96}");
#endif
    }

    if (split_q > 1 && grad_key_feat.defined() && grad_key_feat.numel() > 0) {
        grad_key_feat = grad_key_feat.sum(0);
    }

    if (split_k > 1) {
        if (grad_point_kfeat.defined() && grad_point_kfeat.numel() > 0) grad_point_kfeat = grad_point_kfeat.sum(0);
        if (grad_point_vfeat.defined() && grad_point_vfeat.numel() > 0) grad_point_vfeat = grad_point_vfeat.sum(0);
    } else {
        if (grad_point_kfeat.defined() && grad_point_kfeat.numel() > 0) grad_point_kfeat = grad_point_kfeat.squeeze(0);
        if (grad_point_vfeat.defined() && grad_point_vfeat.numel() > 0) grad_point_vfeat = grad_point_vfeat.squeeze(0);
    }
    if (split_w > 1) {
        if (grad_weights.defined() && grad_weights.numel() > 0) grad_weights = grad_weights.sum(0);
        if (grad_bias.defined() && grad_bias.numel() > 0) grad_bias = grad_bias.sum(0);
    } else {
        if (grad_weights.defined() && grad_weights.numel() > 0) grad_weights = grad_weights.squeeze(0);
        if (grad_bias.defined() && grad_bias.numel() > 0) grad_bias = grad_bias.squeeze(0);
    }


    return {grad_key_feat, grad_point_kfeat, grad_point_vfeat, grad_weights, grad_bias};
}
