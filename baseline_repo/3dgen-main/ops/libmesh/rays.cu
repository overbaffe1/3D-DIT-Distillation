#include <cutil_math.h>
#include <mesh.h>
#include <bvh.h>


#define STACK_SIZE 128
#define MAX_DEPTH 48


// __device__ __forceinline__ bool find_ray_tri_point(
//     const float3 &ray_o, const float3 &ray_d,
//     const float3 &v0, const float3 &v1, const float3 &v2,
//     float &dist, float &u, float &v
// ){

//     constexpr float BARY_EPSILON = 1e-4f;

//     const float3 edge0 = v1 - v0;
//     const float3 edge1 = v2 - v0;

//     float3 pvec = cross(ray_d, edge1);
//     double det = static_cast<double>(edge0.x) * pvec.x + static_cast<double>(edge0.y) * pvec.y + static_cast<double>(edge0.z) * pvec.z;
//     if (fabs(det) < 1e-12) return false;
//     double invdet = 1.0 / det;

//     float3 tvec = ray_o - v0;
//     float3 qvec = cross(tvec, edge0);

//     u = static_cast<float>((static_cast<double>(tvec.x) * pvec.x + static_cast<double>(tvec.y) * pvec.y + static_cast<double>(tvec.z) * pvec.z) * invdet);
//     if (u < -BARY_EPSILON || u > 1.0f + BARY_EPSILON) return false;

//     v = static_cast<float>((static_cast<double>(ray_d.x) * qvec.x + static_cast<double>(ray_d.y) * qvec.y + static_cast<double>(ray_d.z) * qvec.z) * invdet);
//     if (v < -BARY_EPSILON || u + v > 1.0f + BARY_EPSILON) return false;

//     dist = static_cast<float>((static_cast<double>(edge1.x) * qvec.x + static_cast<double>(edge1.y) * qvec.y + static_cast<double>(edge1.z) * qvec.z) * invdet); 
//     if(dist < 1e-12f) return false;

//     u = fminf(fmaxf(u, 0.0f), 1.0f);
//     v = fminf(fmaxf(v, 0.0f), 1.0f);
//     dist = fmaxf(dist, 0.0f);

//     return true;
// }

__device__ __forceinline__ bool find_ray_tri_point(
    const float3 &ray_o, const float3 &ray_d,
    const float3 &v0, const float3 &v1, const float3 &v2,
    float &dist, float &u, float &v
){

    // reindexing coordinates for stability
    const float ax = fabsf(ray_d.x), ay = fabsf(ray_d.y), az = fabsf(ray_d.z);
    int kz = (az > ay) ? (az > ax ? 2 : 0) : (ay > ax ? 1 : 0);
    int kx, ky;
    if ((&ray_d.x)[kz] > 0.0f){
        kx = (kz + 1) % 3; ky = (kz + 2) % 3;
    } else {
        kx = (kz + 2) % 3; ky = (kz + 1) % 3;
    }

    // compute transform matrix coefficients
    const float Sz = 1.0f / (&ray_d.x)[kz];
    const float Sx = (&ray_d.x)[kx] * Sz;
    const float Sy = (&ray_d.x)[ky] * Sz; 
    
    // relative coords
    const float3 A = v0 - ray_o, B = v1 - ray_o, C = v2 - ray_o;

    // projected coords
    const float Ax = (&A.x)[kx] - Sx * (&A.x)[kz];
    const float Ay = (&A.x)[ky] - Sy * (&A.x)[kz];
    const float Bx = (&B.x)[kx] - Sx * (&B.x)[kz];
    const float By = (&B.x)[ky] - Sy * (&B.x)[kz];
    const float Cx = (&C.x)[kx] - Sx * (&C.x)[kz];
    const float Cy = (&C.x)[ky] - Sy * (&C.x)[kz];

    // unnormalized barycentrics
    const double U = static_cast<double>(Cx) * By - static_cast<double>(Cy) * Bx;
    const double V = static_cast<double>(Ax) * Cy - static_cast<double>(Ay) * Cx;
    const double W = static_cast<double>(Bx) * Ay - static_cast<double>(By) * Ax;
    if ((U < 0.0 || V < 0.0 || W < 0.0) && (U > 0.0 || V > 0.0 || W > 0.0)) return false;

    const double det = U + V + W;
    if (fabs(det) < 1e-12) return false;
    const double invdet = 1.0 / det;

    const float Az = (&A.x)[kz] * Sz;
    const float Bz = (&B.x)[kz] * Sz;
    const float Cz = (&C.x)[kz] * Sz;
    dist = static_cast<float>((U * Az + V * Bz + W * Cz) * invdet);
    if (dist < 1e-12f) return false;

    dist = fmaxf(dist, 0.0f);
    u = fminf(fmaxf(static_cast<float>(V * invdet), 0.0f), 1.0f);
    v = fminf(fmaxf(static_cast<float>(W * invdet), 0.0f), 1.0f);

    return true;

}


__device__  bool ray_bbox_bounded(const float3 &p, const float3 &d, const float3 &bb_min_orig, const float3 &bb_max_orig, float ext, float max_dist){
    const float3 bb_min = bb_min_orig - ext;
    const float3 bb_max = bb_max_orig + ext;

    float txmin, txmax, tymin, tymax, tzmin, tzmax;

    float divx = 1 / d.x;
    if(divx >= 0){
        txmin = (bb_min.x - p.x) * divx;
        txmax = (bb_max.x - p.x) * divx;
    } else {
        txmin = (bb_max.x - p.x) * divx;
        txmax = (bb_min.x - p.x) * divx;
    }

    float divy = 1 / d.y;
    if(divy >= 0){
        tymin = (bb_min.y - p.y) * divy;
        tymax = (bb_max.y - p.y) * divy;
    } else {
        tymin = (bb_max.y - p.y) * divy;
        tymax = (bb_min.y - p.y) * divy;
    }

    float divz = 1 / d.z;
    if(divz >= 0){
        tzmin = (bb_min.z - p.z) * divz;
        tzmax = (bb_max.z - p.z) * divz;
    } else {
        tzmin = (bb_max.z - p.z) * divz;
        tzmax = (bb_min.z - p.z) * divz;
    }


    if((txmin > tymax) || (tymin > txmax)) return false;

    if(tymin > txmin) txmin = tymin;
    if(tymax < txmax) txmax = tymax;
    if((txmin > tzmax) || (tzmin > txmax)) return false;

    if(tzmin > txmin) txmin = tzmin;
    if(tzmax < txmax) txmax = tzmax;

    return (txmax >= 0.0f) && (txmin <= max_dist);

}


template <int n_layers>
__device__ void bvh_ray_intersect(
    const float3 &ori, const float3 &dir,
    const int3* __restrict__ nodes, const int2* __restrict__ ranges,
    const float3* __restrict__ bbox, const float3* __restrict__ tri,
    int* __restrict__  min_tri, float* __restrict__ min_depth,
    float* __restrict__ min_u, float* __restrict__ min_v
) {

    constexpr float DEPTH_EPSILON = 1e-6f;

    int64_t stack[STACK_SIZE]; stack[0] = 0; int stack_size = 1;
    int node = 0, tree_depth = 0;
    do {

        const int3 childs = nodes[node];
        const int left_index = childs.x, right_index = childs.y;

        bool traverseL = false;
        if (left_index >= 0 && tree_depth < MAX_DEPTH){
            const float3 bbox_min_left = bbox[left_index * 2 + 0];
            const float3 bbox_max_left = bbox[left_index * 2 + 1];
            traverseL = ray_bbox_bounded(ori, dir, bbox_min_left, bbox_max_left, 1e-6f, min_depth[n_layers - 1]);
        }

        bool traverseR = false;
        if(right_index >= 0 && tree_depth < MAX_DEPTH){
            const float3 bbox_min_right = bbox[right_index * 2 + 0];
            const float3 bbox_max_right = bbox[right_index * 2 + 1];
            traverseR = ray_bbox_bounded(ori, dir, bbox_min_right, bbox_max_right, 1e-6f, min_depth[n_layers - 1]);
        }

        if ((left_index < 0 && right_index < 0) || (tree_depth >= MAX_DEPTH)) {

            // LEAF
            const int2 bounds = ranges[node];
            const int left_bound = bounds.x, right_bound = bounds.y;

            for(int tri_idx = left_bound; tri_idx <= right_bound; tri_idx++){
                const float3* curr_triangle = tri + tri_idx * 3;
                const float3 v0 = curr_triangle[0], v1 = curr_triangle[1], v2 = curr_triangle[2];

                // try to found triangle intersection
                float tri_depth, tri_u, tri_v;
                if(!find_ray_tri_point(ori, dir, v0, v1, v2, tri_depth, tri_u, tri_v)) continue;

                // Insert if strictly closer or nearly equal depth but smaller triangle id
                if ((tri_depth + DEPTH_EPSILON) < min_depth[n_layers - 1] || (fabsf(tri_depth - min_depth[n_layers - 1]) <= DEPTH_EPSILON && tri_idx < min_tri[n_layers - 1])){

                    min_tri[n_layers - 1] = tri_idx; min_depth[n_layers - 1] = tri_depth;
                    min_u[n_layers - 1] = tri_u; min_v[n_layers - 1] = tri_v;

                    // restore order
                    for (int k = n_layers - 2; k >= 0
                         && ((min_depth[k + 1] + DEPTH_EPSILON < min_depth[k]) || (fabsf(min_depth[k] - min_depth[k + 1]) <= DEPTH_EPSILON && min_tri[k] > min_tri[k + 1])); --k)
                    {

                        int itmp = min_tri[k]; min_tri[k] = min_tri[k + 1]; min_tri[k + 1] = itmp;
                        float ftmp;
                        ftmp = min_depth[k]; min_depth[k] = min_depth[k + 1]; min_depth[k + 1] = ftmp;
                        ftmp = min_u[k]; min_u[k] = min_u[k + 1]; min_u[k + 1] = ftmp;
                        ftmp = min_v[k]; min_v[k] = min_v[k + 1]; min_v[k + 1] = ftmp;
                    }
                }
            }
        }

        if (!traverseL && !traverseR){
            int64_t item = stack[--stack_size];
            node = item >> 8; tree_depth = item & 0xFFLL;
        }
        else {
            node = traverseL ? left_index : right_index;
            tree_depth++;
            if(traverseL && traverseR && stack_size < STACK_SIZE) {
                stack[stack_size++] = (static_cast<int64_t>(right_index) << 8) | static_cast<int64_t>(tree_depth);
            }
        }

    } while(stack_size > 0);

}



struct pcg32 {

    static constexpr uint64_t PCG32_MULT = 0x5851f42d4c957f2dULL;

	__device__ pcg32(uint64_t initstate) { seed(initstate); }

	__device__ void seed(uint64_t initstate, uint64_t initseq = 1u) {
		state = 0U;
		inc = (initseq << 1u) | 1u;
		next_uint();
		state += initstate;
		next_uint();
	}

	__device__ uint32_t next_uint() {
		uint64_t oldstate = state;
		state = oldstate * PCG32_MULT + inc;
		uint32_t xorshifted = (uint32_t) (((oldstate >> 18u) ^ oldstate) >> 27u);
		uint32_t rot = (uint32_t) (oldstate >> 59u);
		return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
	}

	__device__ uint32_t next_uint(uint32_t bound) {
		uint32_t threshold = (~bound+1u) % bound;
		for (;;) {
			uint32_t r = next_uint();
			if (r >= threshold)
				return r % bound;
		}
	}

	__device__ float next_float() {
		union {
			uint32_t u;
			float f;
		} x;
		x.u = (next_uint() >> 9) | 0x3f800000u;
		return x.f - 1.0f;
	}

	uint64_t state;
	uint64_t inc;
};

__device__ float3 fibonacci_dir(uint32_t i, const float off_x, const float off_y, const int N_DIRS) {

    static constexpr float PI = 3.14159265358979323846f;
	static constexpr float GOLDEN_RATIO = 1.6180339887498948482045868343656f;

	float epsilon;
	if (N_DIRS >= 11000) {
		epsilon = 27;
	} else if (N_DIRS >= 890) {
		epsilon = 10;
	} else if (N_DIRS >= 177) {
		epsilon = 3.33;
	} else if (N_DIRS >= 24) {
		epsilon = 1.33;
	} else {
		epsilon = 0.33;
	}

	float px = (i+epsilon) / (N_DIRS-1+2*epsilon) + off_x;
	px -= floorf(px);

    const float cos_theta = -2.0f * px + 1.0f;
	const float sin_theta = sqrtf(fmaxf(1.0f - cos_theta * cos_theta, 0.0f));

	float py = i / GOLDEN_RATIO + off_y;
	py -= floorf(py);

	const float phi = 2.0f * PI * (py - 0.5f);
	float sin_phi, cos_phi;
	sincosf(phi, &sin_phi, &cos_phi);

	return make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
}

__global__ void orient_rays_kernel(
    const int N_TRI,
    const int N_RAYS,
    const int N_BOUNCE,
    const float3* __restrict__ tri,
    const int3* __restrict__ nodes,
    const int2* __restrict__ ranges,
    const float3* __restrict__ bbox,
    const float3* __restrict__ sorted_tri,
    int2* __restrict__ ray_cnt
){
    const int gidx = global_idx();
    if (gidx >= N_TRI) return;

    const float3 v0 = tri[gidx * 3 + 0], v1 = tri[gidx * 3 + 1], v2 = tri[gidx * 3 + 2];
    const float3 nrm = normalize(cross(v1 - v0, v2 - v0));

    pcg32 rng(static_cast<uint64_t>(41 + gidx));
    const float off_x = rng.next_float(), off_y = rng.next_float();

    int survived[2] = {0, 0};

    for(int ray_idx = 0; ray_idx < N_RAYS * 2; ray_idx++){

        float3 dir = normalize(fibonacci_dir(ray_idx % N_RAYS, off_x, off_y, N_RAYS));
        if ((ray_idx < N_RAYS && dot(dir, nrm) < 0.0f) || (ray_idx >= N_RAYS && dot(dir, nrm) > 0.0f)) dir = -dir;

        float uu = sqrtf(clamp(rng.next_float(), 0.0f, 1.0f)), vv = rng.next_float();
        float3 ori = (1 - uu) * v0 + uu * (1 - vv) * v1 + uu * vv * v2;
        ori += dir * 1e-6f;

        for(int bounce_idx = 0; bounce_idx < N_BOUNCE; bounce_idx++){

            int found_idx = -1; float found_depth = 1e6f, found_u = 0.0f, found_v = 0.0f;
            bvh_ray_intersect<1>(ori, dir, nodes, ranges, bbox, sorted_tri, &found_idx, &found_depth, &found_u, &found_v);
            if (found_idx < 0) {
                survived[ray_idx / N_RAYS]++;
                break;
            }

            // rebuild dir from new triangle
            float3 nxt_v0 = sorted_tri[found_idx * 3 + 0], nxt_v1 = sorted_tri[found_idx * 3 + 1], nxt_v2 = sorted_tri[found_idx * 3 + 2];
            float3 nxt_nrm = normalize(cross(nxt_v1 - nxt_v0, nxt_v2 - nxt_v0));
            if(dot(-dir, nxt_nrm) < 0.0f) nxt_nrm = -nxt_nrm;

            dir = normalize(2.0f * dot(-dir, nxt_nrm) * nxt_nrm + dir); // reflect along nxt_nrm

            ori = nxt_v0 * (1 - found_u - found_v) + nxt_v1 * found_u + nxt_v2 * found_v;
            ori += nxt_nrm * 1e-6f;

        }

    }

    ray_cnt[gidx] = make_int2(survived[0], survived[1]);
}

torch::Tensor cuda_orient_rays(torch::Tensor qtri, torch::Tensor tri, int n_rays, int n_bounce){

    const int threads = 128;
    auto device = tri.device();

    bvh tree = build_tri_bvh(tri);
    auto sorted_tri = tri.index({tree.orig_idx});

    const int N_TRI = qtri.size(0);
    auto cnt = torch::empty({N_TRI, 2}, torch::TensorOptions().device(device).dtype(torch::kInt32));

    orient_rays_kernel<<<(N_TRI + threads - 1) / threads, threads>>>(
        N_TRI,
        n_rays,
        n_bounce,
        reinterpret_cast<float3*>(qtri.data_ptr<float>()),
        reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
        reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
        reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
        reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
        reinterpret_cast<int2*>(cnt.data_ptr<int>())
    );

    return cnt;
}


__device__ __forceinline__ uint32_t part1by1_u32(uint32_t v) {
    // Spread lower 12 bits to even positions in 32-bit lane.
    v &= 0x00000FFFu;
    v = (v | (v << 8))  & 0x00FF00FFu;
    v = (v | (v << 4))  & 0x0F0F0F0Fu;
    v = (v | (v << 2))  & 0x33333333u;
    v = (v | (v << 1))  & 0x55555555u;
    return v;
}

__device__ __forceinline__ uint32_t morton2d_u32(uint32_t x, uint32_t y) {
    return part1by1_u32(x << 1) | part1by1_u32(y);
}

__global__ void morton_2d_kernel(
    const int N_RAYS,
    const int3* __restrict__ coords,
    int* __restrict__ code
) {

    const int gidx = global_idx();
    if (gidx >= N_RAYS) return;

    const int3 coord = coords[gidx];
    uint32_t b = static_cast<uint32_t>(coord.x);
    uint32_t x = static_cast<uint32_t>(coord.y);
    uint32_t y = static_cast<uint32_t>(coord.z);

    code[gidx] = static_cast<int32_t>((b << 24u) | morton2d_u32(x, y));
}


template <int n_layers>
__global__ void cast_rays_kernel(
    const int N_RAYS,
    const float3* __restrict__ ray_ori,
    const float3* __restrict__ ray_dir,
    const int3* __restrict__ nodes,
    const int2* __restrict__ ranges,
    const float3* __restrict__ bbox,
    const int64_t* __restrict__ orig_idx,
    const float3* __restrict__ tri,
    int* __restrict__ found_idx,
    float* __restrict__ depth,
    float2* __restrict__ uv
) {

    const int gidx = global_idx();
    if (gidx >= N_RAYS) return;

    const float3 ori = ray_ori[gidx];
    const float3 dir_orig = ray_dir[gidx];
    const float3 dir = normalize(dir_orig);

    int min_tri[n_layers]; float min_depth[n_layers], min_u[n_layers], min_v[n_layers];
    #pragma unroll
    for (size_t k = 0; k < n_layers; k++)
    {
        min_tri[k] = -1;
        min_depth[k] = 1e6f;
        min_u[k] = min_v[k] = 0.0f;
    }

    bvh_ray_intersect<n_layers>(
        ori, dir, nodes, ranges, bbox, tri,
        min_tri, min_depth, min_u, min_v
    );

    #pragma unroll
    for (size_t k = 0; k < n_layers; ++k)
    {
        if (min_tri[k] >= 0) {
            min_tri[k] = static_cast<int32_t>(orig_idx[min_tri[k]]);
            min_depth[k] = min_depth[k] / length(dir_orig);
        }

        found_idx[gidx] = min_tri[k];
        found_idx += N_RAYS;

        depth[gidx] = min_depth[k];
        depth += N_RAYS;

        uv[gidx] = make_float2(min_u[k], min_v[k]);
        uv += N_RAYS;

    }
}

cast_out cuda_cast_rays(
    torch::Tensor tri,
    torch::Tensor ray_ori, torch::Tensor ray_dir, torch::Tensor coords,
    const int N_LAYERS
){

    const int threads = 128;
    auto device = tri.device();

    bvh tree = build_tri_bvh(tri);
    auto sorted_tri = tri.index({tree.orig_idx});

    const int N_RAYS = ray_ori.size(0);
    auto ray_code = torch::empty({N_RAYS}, torch::TensorOptions().device(device).dtype(torch::kInt32));
    morton_2d_kernel<<<(N_RAYS + threads - 1) / threads, threads>>>(
        N_RAYS,
        reinterpret_cast<int3*>(coords.data_ptr<int>()),
        ray_code.data_ptr<int>()
    );

    auto ray_order = torch::argsort(ray_code);
    ray_ori = ray_ori.index({ray_order}); ray_dir = ray_dir.index({ray_order});

    auto idx = torch::full({N_LAYERS, N_RAYS}, -1, torch::TensorOptions().device(device).dtype(torch::kInt32));
    auto depth = torch::full({N_LAYERS, N_RAYS}, 1e6f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    auto uv = torch::full({N_LAYERS, N_RAYS, 2}, 0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    if (N_LAYERS == 4)
    {
        cast_rays_kernel<4><<<(N_RAYS + threads - 1) / threads, threads>>>(
            N_RAYS,
            reinterpret_cast<float3*>(ray_ori.data_ptr<float>()),
            reinterpret_cast<float3*>(ray_dir.data_ptr<float>()),
            reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
            reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
            reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
            tree.orig_idx.data_ptr<int64_t>(),
            reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
            idx.data_ptr<int>(),
            depth.data_ptr<float>(),
            reinterpret_cast<float2*>(uv.data_ptr<float>())
        );
    }
    else if (N_LAYERS == 3)
    {
        cast_rays_kernel<3><<<(N_RAYS + threads - 1) / threads, threads>>>(
            N_RAYS,
            reinterpret_cast<float3*>(ray_ori.data_ptr<float>()),
            reinterpret_cast<float3*>(ray_dir.data_ptr<float>()),
            reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
            reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
            reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
            tree.orig_idx.data_ptr<int64_t>(),
            reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
            idx.data_ptr<int>(),
            depth.data_ptr<float>(),
            reinterpret_cast<float2*>(uv.data_ptr<float>())
        );
    }
    else if (N_LAYERS == 2)
    {
        cast_rays_kernel<2><<<(N_RAYS + threads - 1) / threads, threads>>>(
            N_RAYS,
            reinterpret_cast<float3*>(ray_ori.data_ptr<float>()),
            reinterpret_cast<float3*>(ray_dir.data_ptr<float>()),
            reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
            reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
            reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
            tree.orig_idx.data_ptr<int64_t>(),
            reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
            idx.data_ptr<int>(),
            depth.data_ptr<float>(),
            reinterpret_cast<float2*>(uv.data_ptr<float>())
        );
    }
    else
    {
        cast_rays_kernel<1><<<(N_RAYS + threads - 1) / threads, threads>>>(
            N_RAYS,
            reinterpret_cast<float3*>(ray_ori.data_ptr<float>()),
            reinterpret_cast<float3*>(ray_dir.data_ptr<float>()),
            reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
            reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
            reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
            tree.orig_idx.data_ptr<int64_t>(),
            reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
            idx.data_ptr<int>(),
            depth.data_ptr<float>(),
            reinterpret_cast<float2*>(uv.data_ptr<float>())
        );
    }


    torch::Tensor ray_inverse = torch::empty_like(ray_order);
    ray_inverse.index_put_({ray_order}, torch::arange(0, ray_order.size(0), torch::TensorOptions().dtype(ray_order.dtype()).device(ray_order.device())));

    return {
        idx.index({torch::indexing::Slice(), ray_inverse}),
        depth.index({torch::indexing::Slice(), ray_inverse}),
        uv.index({torch::indexing::Slice(), ray_inverse})
    };
}

// Forward formulas mirrored from find_ray_tri_point (Woop/Wald/Benthin):
// - choose permutation (kz,kx,ky) so |d_kz| is max and flip so d_kz > 0
// - Sx = d_kx / d_kz; Sy = d_ky / d_kz; Sz = 1 / d_kz
// - A = v0 - o; B = v1 - o; C = v2 - o
// - Ax = A_kx - Sx*A_kz; Ay = A_ky - Sy*A_kz  (similarly for Bx,By,Cx,Cy)
// - U = Cx*By - Cy*Bx; V = Ax*Cy - Ay*Cx; W = Bx*Ay - By*Ax; det = U+V+W
// - Az = A_kz*Sz; Bz = B_kz*Sz; Cz = C_kz*Sz
// - dist = (U*Az + V*Bz + W*Cz) / det; u = V / det; v = W / det
// Backward notes:
//   g_det = g_invdet * (-1/det^2)
//   g_U += g_det; g_V += g_det; g_W += g_det
//   U = Cx*By - Cy*Bx -> gCx+=gU*By; gBy+=gU*Cx; gCy+=-gU*Bx; gBx+=-gU*Cy
//   V = Ax*Cy - Ay*Cx -> gAx+=gV*Cy; gCy+=gV*Ax; gAy+=-gV*Cx; gCx+=-gV*Ay
//   W = Bx*Ay - By*Ax -> gBx+=gW*Ay; gAy+=gW*Bx; gBy+=-gW*Ax; gAx+=-gW*By
//   Ax = A_kx - Sx*A_kz -> gA_kx+=gAx; gA_kz+=-gAx*Sx (similarly Ay,Bx,By,Cx,Cy)
//   Az,Bz,Cz scale only g*_kz by Sz
__global__ void cast_rays_bwd_kernel(
    const int N_RAYS,
    const float3* __restrict__ ray_ori,
    const float3* __restrict__ ray_dir,
    const int N_LAYERS,
    const int* __restrict__ tri_idx,
    const float3* __restrict__ tri,
    const float* __restrict__ grad_depth,
    const float* __restrict__ grad_uv,
    float* __restrict__ grad_tri
) {

    const int gidx = global_idx();
    if (gidx >= N_RAYS * N_LAYERS) return;

    const int found_idx = tri_idx[gidx];
    if (found_idx < 0) return;

    const int r_idx = gidx % N_RAYS;
    const float3 ray_o = ray_ori[r_idx];
    const float3 ray_d_orig = ray_dir[r_idx];
    const float3 ray_d = normalize(ray_d_orig);

    const float3* curr_tri = tri + found_idx * 3;
    const float3 v0 = curr_tri[0], v1 = curr_tri[1], v2 = curr_tri[2];

    // ----- Forward recompute -----
    const float ax = fabsf(ray_d.x), ay = fabsf(ray_d.y), az = fabsf(ray_d.z);
    const int kz = (az > ay) ? (az > ax ? 2 : 0) : (ay > ax ? 1 : 0);
    int kx, ky;
    if ((&ray_d.x)[kz] > 0.0f){
        kx = (kz + 1) % 3; ky = (kz + 2) % 3;
    } else {
        kx = (kz + 2) % 3; ky = (kz + 1) % 3;
    }

    const float Sz = 1.0f / (&ray_d.x)[kz];
    const float Sx = (&ray_d.x)[kx] * Sz;
    const float Sy = (&ray_d.x)[ky] * Sz;

    const float3 A = v0 - ray_o, B = v1 - ray_o, C = v2 - ray_o;

    const float Ax = (&A.x)[kx] - Sx * (&A.x)[kz];
    const float Ay = (&A.x)[ky] - Sy * (&A.x)[kz];
    const float Bx = (&B.x)[kx] - Sx * (&B.x)[kz];
    const float By = (&B.x)[ky] - Sy * (&B.x)[kz];
    const float Cx = (&C.x)[kx] - Sx * (&C.x)[kz];
    const float Cy = (&C.x)[ky] - Sy * (&C.x)[kz];

    const double U = static_cast<double>(Cx) * By - static_cast<double>(Cy) * Bx;
    const double V = static_cast<double>(Ax) * Cy - static_cast<double>(Ay) * Cx;
    const double W = static_cast<double>(Bx) * Ay - static_cast<double>(By) * Ax;
    const double det = U + V + W;
    const double invdet = 1.0 / det;

    const float Az = (&A.x)[kz] * Sz;
    const float Bz = (&B.x)[kz] * Sz;
    const float Cz = (&C.x)[kz] * Sz;

    const float dist = static_cast<float>((U * Az + V * Bz + W * Cz) * invdet);
    const float u = static_cast<float>(V * invdet);
    const float v = static_cast<float>(W * invdet);

    const float du = grad_uv[gidx * 2 + 0];
    const float dv = grad_uv[gidx * 2 + 1];
    const float ddist = grad_depth[gidx] / length(ray_d_orig);

    // ----- Reverse pass -----
    double gU = 0.0, gV = 0.0, gW = 0.0;
    double gDet = 0.0, gInvDet = 0.0;
    float gAx = 0.0f, gAy = 0.0f, gBx = 0.0f, gBy = 0.0f, gCx = 0.0f, gCy = 0.0f;
    float gAz = 0.0f, gBz = 0.0f, gCz = 0.0f;
    float gA[3] = {0.0f, 0.0f, 0.0f};
    float gB[3] = {0.0f, 0.0f, 0.0f};
    float gC[3] = {0.0f, 0.0f, 0.0f};

    // dist contribution
    if (dist > 0.0f){
        gInvDet += ddist * (U * Az + V * Bz + W * Cz);

        const double factor = ddist * invdet;
        gU += factor * Az;
        gAz += static_cast<float>(factor * U);
        gV += factor * Bz;
        gBz += static_cast<float>(factor * V);
        gW += factor * Cz;
        gCz += static_cast<float>(factor * W);
    }
    // u contribution
    if (u > 0.0f && u < 1.0f){
        gV += du * invdet;
        gInvDet += du * V;
    }

    // v contribution
    if (v > 0.0f && v < 1.0f){
        gW += dv * invdet;
        gInvDet += dv * W;
    }

    // invdet = 1 / det
    gDet += -1.0 * gInvDet * invdet * invdet;

    // det = U + V + W
    gU += gDet;
    gV += gDet;
    gW += gDet;

    // U = Cx*By - Cy*Bx
    gCx += static_cast<float>(gU) * By;
    gBy += static_cast<float>(gU) * Cx;
    gCy += static_cast<float>(gU) * (-Bx);
    gBx += static_cast<float>(gU) * (-Cy);

    // V = Ax*Cy - Ay*Cx
    gAx += static_cast<float>(gV) * Cy;
    gCy += static_cast<float>(gV) * Ax;
    gAy += static_cast<float>(gV) * (-Cx);
    gCx += static_cast<float>(gV) * (-Ay);

    // W = Bx*Ay - By*Ax
    gBx += static_cast<float>(gW) * Ay;
    gAy += static_cast<float>(gW) * Bx;
    gBy += static_cast<float>(gW) * (-Ax);
    gAx += static_cast<float>(gW) * (-By);

    // Ax = A_kx - Sx * A_kz
    gA[kx] += gAx;
    gA[kz] += gAx * (-Sx);

    // Ay = A_ky - Sy * A_kz
    gA[ky] += gAy;
    gA[kz] += gAy * (-Sy);

    // Bx, By
    gB[kx] += gBx;
    gB[kz] += gBx * (-Sx);
    gB[ky] += gBy;
    gB[kz] += gBy * (-Sy);

    // Cx, Cy
    gC[kx] += gCx;
    gC[kz] += gCx * (-Sx);
    gC[ky] += gCy;
    gC[kz] += gCy * (-Sy);

    // Az = A_kz * Sz, etc. (Sz treated constant w.r.t vertices)
    gA[kz] += gAz * Sz;
    gB[kz] += gBz * Sz;
    gC[kz] += gCz * Sz;

    float3 v0_grad = make_float3(gA[0], gA[1], gA[2]);
    float3 v1_grad = make_float3(gB[0], gB[1], gB[2]);
    float3 v2_grad = make_float3(gC[0], gC[1], gC[2]);

    float* curr_grad_tri = grad_tri + 9 * found_idx;
    atomicAdd(curr_grad_tri + 3 * 0 + 0, v0_grad.x); atomicAdd(curr_grad_tri + 3 * 0 + 1, v0_grad.y); atomicAdd(curr_grad_tri + 3 * 0 + 2, v0_grad.z);
    atomicAdd(curr_grad_tri + 3 * 1 + 0, v1_grad.x); atomicAdd(curr_grad_tri + 3 * 1 + 1, v1_grad.y); atomicAdd(curr_grad_tri + 3 * 1 + 2, v1_grad.z);
    atomicAdd(curr_grad_tri + 3 * 2 + 0, v2_grad.x); atomicAdd(curr_grad_tri + 3 * 2 + 1, v2_grad.y); atomicAdd(curr_grad_tri + 3 * 2 + 2, v2_grad.z);

}


torch::Tensor cuda_cast_rays_bwd(
    torch::Tensor tri,
    torch::Tensor ray_ori, torch::Tensor ray_dir, const int N_LAYERS,
    torch::Tensor tri_idx,
    torch::Tensor grad_depth, torch::Tensor grad_uv
) {
    constexpr int threads = 256;
    auto device = tri.device();

    const int N_TRI = tri.size(0), N_RAYS = ray_ori.size(0);
    auto grad_tri = torch::zeros({N_TRI, 3, 3}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    cast_rays_bwd_kernel<<<(N_RAYS * N_LAYERS + threads - 1) / threads, threads>>>(
        N_RAYS,
        reinterpret_cast<float3*>(ray_ori.data_ptr<float>()),
        reinterpret_cast<float3*>(ray_dir.data_ptr<float>()),
        N_LAYERS,
        tri_idx.data_ptr<int>(),
        reinterpret_cast<float3*>(tri.data_ptr<float>()),
        grad_depth.data_ptr<float>(),
        grad_uv.data_ptr<float>(),
        grad_tri.data_ptr<float>()
    );

    return grad_tri;
}


__device__ float point_to_segment_dist_sq_2d(
    const float2 &p,
    const float2 &a,
    const float2 &b,
    float &t
){

    const float2 ba = b - a;
    const double len2 = static_cast<double>(ba.x) * ba.x + static_cast<double>(ba.y) * ba.y;

    if (len2 > 1e-12) {
        const float2 pa = p - a;
        t = fminf(fmaxf(static_cast<float>((static_cast<double>(pa.x) * ba.x + static_cast<double>(pa.y) * ba.y) / len2), 0.0f), 1.0f);
    } else {
        t = 0.0f;
    }

    const float2 q = a + ba * t;
    const float2 dx = p - q;
    return static_cast<float>(sqrt(static_cast<double>(dx.x) * dx.x + static_cast<double>(dx.y) * dx.y));
}

__device__ __forceinline__ double cross2d(const float2 &a, const float2 &b) {
    return static_cast<double>(a.x) * b.y - static_cast<double>(a.y) * b.x;
}

__device__ bool ray_tri_proximity(
    const float3 &ray_o, const float3 &ray_d,
    const float3 &v0, const float3 &v1, const float3 &v2,
    const float z_far, const float radius,
    float &depth, float &d, float3 &q,
    float &bary_u, float &bary_v
){

    const float3 w = ray_d;
    float3 up = make_float3(0.0f, 1.0f, 0.0f);
    if (fabsf(w.y) > 0.99f) up = make_float3(1.0f, 0.0f, 0.0f);
    const float3 u = normalize(cross(w, up));
    const float3 v = normalize(cross(w, u));

    const float3 p0 = v0 - ray_o;
    const float3 p1 = v1 - ray_o;
    const float3 p2 = v2 - ray_o;

    const float2 p0xy = make_float2(dot(p0, u), dot(p0, v));
    const float2 p1xy = make_float2(dot(p1, u), dot(p1, v));
    const float2 p2xy = make_float2(dot(p2, u), dot(p2, v));

    const float2 origin = make_float2(0.0f, 0.0f);
    const double w0 = cross2d(p1xy - origin, p2xy - origin);
    const double w1 = cross2d(p2xy - origin, p0xy - origin);
    const double w2 = cross2d(p0xy - origin, p1xy - origin);

    const bool inside = (w0 > 0.0 && w1 > 0.0 && w2 > 0.0) || (w0 < 0.0 && w1 < 0.0 && w2 < 0.0);

    float t01, t12, t20;
    const float d01 = point_to_segment_dist_sq_2d(origin, p0xy, p1xy, t01);
    const float d12 = point_to_segment_dist_sq_2d(origin, p1xy, p2xy, t12);
    const float d20 = point_to_segment_dist_sq_2d(origin, p2xy, p0xy, t20);

    float b0, b1, b2;
    if (inside){
        d = -fminf(d01, fminf(d12, d20));
        double invarea = 1.0 / (w0 + w1 + w2 + 1e-12);
        b0 = fminf(fmaxf(static_cast<float>(w0 * invarea), 0.0f), 1.0f);
        b1 = fminf(fmaxf(static_cast<float>(w1 * invarea), 0.0f), 1.0f);
        b2 = fminf(fmaxf(static_cast<float>(w2 * invarea), 0.0f), 1.0f);
    } else {
        d = d01; b0 = (1.0f - t01); b1 = t01; b2 = 0.0f;
        if (d12 < d){
            d = d12; b0 = 0.0f; b1 = (1.0f - t12); b2 = t12;
        }
        if (d20 < d){
            d = d20; b0 = t20; b1 = 0.0f; b2 = (1.0f - t20);
        }
    }

    if (!inside && d > radius) return false;

    q = b0 * v0 + b1 * v1 + b2 * v2;
    depth = dot(q - ray_o, w);
    if (depth <= 0.0f || depth >= z_far) return false;

    bary_u = b1;
    bary_v = b2;

    return true;
}

__device__ void bvh_radius_ray_intersect(
    const float3 &ori, const float3 &dir,
    const int3* __restrict__ nodes, const int2* __restrict__ ranges,
    const float3* __restrict__ bbox, const float3* __restrict__ tri,
    const int K, const float z_far,
    int* __restrict__  min_tri, float* __restrict__ min_depth, float* __restrict__ min_radius,
    float* __restrict__ min_bary_u, float* __restrict__ min_bary_v
) {
    int64_t stack[STACK_SIZE]; stack[0] = 0; int stack_size = 1;
    int node = 0, tree_depth = 0;

    do {
        const int3 childs = nodes[node];
        const int left_index = childs.x, right_index = childs.y;

        bool traverseL = false;
        if (left_index >= 0 && tree_depth < MAX_DEPTH){
            const float3 bbox_min_left = bbox[left_index * 2 + 0];
            const float3 bbox_max_left = bbox[left_index * 2 + 1];
            traverseL = ray_bbox_bounded(ori, dir, bbox_min_left, bbox_max_left, fmaxf(min_radius[K - 1], 1e-6f), z_far);
        }

        bool traverseR = false;
        if(right_index >= 0 && tree_depth < MAX_DEPTH){
            const float3 bbox_min_right = bbox[right_index * 2 + 0];
            const float3 bbox_max_right = bbox[right_index * 2 + 1];
            traverseR = ray_bbox_bounded(ori, dir, bbox_min_right, bbox_max_right, fmaxf(min_radius[K - 1], 1e-6f), z_far);
        }

        if ((left_index < 0 && right_index < 0) || (tree_depth == MAX_DEPTH)) {
            // LEAF
            const int2 bounds = ranges[node];
            const int left_bound = bounds.x, right_bound = bounds.y;

            for(int tri_idx = left_bound; tri_idx <= right_bound; tri_idx++){
                const float3* curr_triangle = tri + tri_idx * 3;
                const float3 v0 = curr_triangle[0], v1 = curr_triangle[1], v2 = curr_triangle[2];

                float d; float3 q; float depth; float bary_u; float bary_v;
                if(!ray_tri_proximity(ori, dir, v0, v1, v2, z_far, min_radius[K - 1], depth, d, q, bary_u, bary_v)) continue;

                if (d < min_radius[K - 1]){
                    // Insert into sorted array
                    min_tri[K - 1] = tri_idx;
                    min_depth[K - 1] = depth;
                    min_radius[K - 1] = d;
                    min_bary_u[K - 1] = bary_u;
                    min_bary_v[K - 1] = bary_v;

                    int j = K - 1;
                    while (j > 0 && min_radius[j - 1] > min_radius[j])
                    {
                        int itmp = min_tri[j - 1]; min_tri[j - 1] = min_tri[j]; min_tri[j] = itmp;
                        float dtmp = min_depth[j - 1]; min_depth[j - 1] = min_depth[j]; min_depth[j] = dtmp;
                        float rtmp = min_radius[j - 1]; min_radius[j - 1] = min_radius[j]; min_radius[j] = rtmp;
                        float utmp = min_bary_u[j - 1]; min_bary_u[j - 1] = min_bary_u[j]; min_bary_u[j] = utmp;
                        float vtmp = min_bary_v[j - 1]; min_bary_v[j - 1] = min_bary_v[j]; min_bary_v[j] = vtmp;
                    }
                }
            }
        }

        if (!traverseL && !traverseR){
            int64_t item = stack[--stack_size];
            node = item >> 8; tree_depth = item & 0xFFLL;
        }
        else {
            node = traverseL ? left_index : right_index;
            tree_depth++;
            if(traverseL && traverseR && stack_size < STACK_SIZE) {
                stack[stack_size++] = (static_cast<int64_t>(right_index) << 8) | static_cast<int64_t>(tree_depth);
            }
        }

    } while(stack_size > 0);

}

template <int max_triangles>
__global__ void cast_radius_rays_kernel(
    const int N_RAYS, const int N_CANDIDATES,
    const float3* __restrict__ ray_ori,
    const float3* __restrict__ ray_dir,
    const int3* __restrict__ nodes,
    const int2* __restrict__ ranges,
    const float3* __restrict__ bbox,
    const int64_t* __restrict__ orig_idx,
    const float3* __restrict__ tri,
    int* __restrict__ found_idx,
    int* __restrict__ hard_mask,
    float* __restrict__ out_depth,
    float2* __restrict__ out_uv,
    float* __restrict__ out_alpha,
    const float radius,
    const float sigma
) {
    const int gidx = global_idx();
    if (gidx >= N_RAYS) return;

    const float3 ori = ray_ori[gidx];
    const float3 dir_orig = ray_dir[gidx];
    const float3 dir = normalize(dir_orig);

    // find first hard intersection
    int hard_tri = -1; float hard_depth = 1e6f; float hard_u = 0.0f, hard_v = 0.0f;
    bvh_ray_intersect<1>(
        ori, dir,
        nodes, ranges, bbox, tri,
        &hard_tri, &hard_depth, &hard_u, &hard_v
    );
    if (hard_tri >= 0) {
        hard_mask[gidx] = static_cast<int32_t>(orig_idx[hard_tri]);
    }

    // init arrays for remained candidates (hard intersection would be the best)
    int min_tri[max_triangles]; float min_depth[max_triangles]; float min_radius[max_triangles];
    float min_bary_u[max_triangles]; float min_bary_v[max_triangles];

    #pragma unroll
    for (size_t k = 0; k < max_triangles; k++) {
        min_tri[k] = -1; min_depth[k] = 1e6f; min_radius[k] = radius + FLT_EPSILON;
        min_bary_u[k] = min_bary_v[k] = 0.0f;
    }
    if (hard_tri >= 0) {
        min_tri[0] = hard_tri; min_depth[0] = hard_depth; min_radius[0] = -1e6f;
        min_bary_u[0] = hard_u; min_bary_v[0] = hard_v;
    }

    // take best triangles by 2D distance to the ray
    const int N_REMAINED = hard_tri >= 0 ? N_CANDIDATES - 1 : N_CANDIDATES;
    if (N_REMAINED > 0){
        bvh_radius_ray_intersect(
            ori, dir,
            nodes, ranges, bbox, tri,
            N_REMAINED, hard_depth,
            min_tri, min_depth, min_radius,
            min_bary_u, min_bary_v
        );
    }

    // resort arrays by depth (min_dist not used so keep as is)
    for (int i = 1; i < N_CANDIDATES; ++i)
    {
        int j = i;
        while (j > 0 && min_depth[j - 1] > min_depth[j])
        {
            int itmp = min_tri[j - 1]; min_tri[j - 1] = min_tri[j]; min_tri[j] = itmp;
            float ftmp = min_depth[j - 1]; min_depth[j - 1] = min_depth[j]; min_depth[j] = ftmp;
            float rtmp = min_radius[j - 1]; min_radius[j - 1] = min_radius[j]; min_radius[j] = rtmp;
            float utmp = min_bary_u[j - 1]; min_bary_u[j - 1] = min_bary_u[j]; min_bary_u[j] = utmp;
            float vtmp = min_bary_v[j - 1]; min_bary_v[j - 1] = min_bary_v[j]; min_bary_v[j] = vtmp;
        }
    }

    // write top 2D dist found triangles in depth order
    for (size_t k = 0; k < N_CANDIDATES; ++k) {
        int final_idx = -1;
        if (min_tri[k] >= 0) final_idx = static_cast<int32_t>(orig_idx[min_tri[k]]);

        found_idx[gidx + k * N_RAYS] = final_idx;
        out_depth[gidx + k * N_RAYS] = min_depth[k] / length(dir_orig);
        out_uv[gidx + k * N_RAYS] = make_float2(min_bary_u[k], min_bary_v[k]);

        float alpha = 0.0f;
        if (min_tri[k] >= 0) {
            if (hard_tri >= 0 && min_tri[k] == hard_tri) {
                alpha = 1.0f;
            } else {
                float d = min_radius[k];
                // alpha = expf(-fmaxf(d, 0.0f) / sigma);
                alpha = 1.0f - fminf(1.0f, fmaxf(d, 0.0f) / sigma);
            }
        }
        out_alpha[gidx + k * N_RAYS] = alpha;
    }
}

radius_out cuda_cast_radius_rays(
    torch::Tensor tri,
    torch::Tensor ray_ori,
    torch::Tensor ray_dir,
    torch::Tensor coords,
    const int N_CANDIDATES,
    float radius,
    float sigma
) {
    const int threads = 128;
    auto device = tri.device();

    bvh tree = build_tri_bvh(tri);
    auto sorted_tri = tri.index({tree.orig_idx});

    const int N_RAYS = ray_ori.size(0);
    auto ray_code = torch::empty({N_RAYS}, torch::TensorOptions().device(device).dtype(torch::kInt32));
    morton_2d_kernel<<<(N_RAYS + threads - 1) / threads, threads>>>(
        N_RAYS,
        reinterpret_cast<int3*>(coords.data_ptr<int>()),
        ray_code.data_ptr<int>()
    );

    auto ray_order = torch::argsort(ray_code);
    ray_ori = ray_ori.index({ray_order});
    ray_dir = ray_dir.index({ray_order});

    auto idx = torch::full({N_CANDIDATES, N_RAYS}, -1, torch::TensorOptions().device(device).dtype(torch::kInt32));
    auto depth = torch::full({N_CANDIDATES, N_RAYS}, 1e6f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    auto uv = torch::full({N_CANDIDATES, N_RAYS, 2}, 0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    auto alpha = torch::zeros({N_CANDIDATES, N_RAYS}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    auto hard_mask = torch::full({N_RAYS}, -1, torch::TensorOptions().device(device).dtype(torch::kInt32));

    if (N_CANDIDATES <= 1) {
        cast_radius_rays_kernel<1><<<(N_RAYS + threads - 1) / threads, threads>>>(
            N_RAYS, N_CANDIDATES,
            reinterpret_cast<float3*>(ray_ori.data_ptr<float>()),
            reinterpret_cast<float3*>(ray_dir.data_ptr<float>()),
            reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
            reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
            reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
            tree.orig_idx.data_ptr<int64_t>(),
            reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
            idx.data_ptr<int>(), hard_mask.data_ptr<int>(),
            depth.data_ptr<float>(), reinterpret_cast<float2*>(uv.data_ptr<float>()), alpha.data_ptr<float>(),
            radius, sigma
        );
    } else if (N_CANDIDATES <= 2) {
        cast_radius_rays_kernel<2><<<(N_RAYS + threads - 1) / threads, threads>>>(
            N_RAYS, N_CANDIDATES,
            reinterpret_cast<float3*>(ray_ori.data_ptr<float>()),
            reinterpret_cast<float3*>(ray_dir.data_ptr<float>()),
            reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
            reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
            reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
            tree.orig_idx.data_ptr<int64_t>(),
            reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
            idx.data_ptr<int>(), hard_mask.data_ptr<int>(),
            depth.data_ptr<float>(), reinterpret_cast<float2*>(uv.data_ptr<float>()), alpha.data_ptr<float>(),
            radius, sigma
        );
    } else if (N_CANDIDATES <= 4) {
        cast_radius_rays_kernel<4><<<(N_RAYS + threads - 1) / threads, threads>>>(
            N_RAYS, N_CANDIDATES,
            reinterpret_cast<float3*>(ray_ori.data_ptr<float>()),
            reinterpret_cast<float3*>(ray_dir.data_ptr<float>()),
            reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
            reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
            reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
            tree.orig_idx.data_ptr<int64_t>(),
            reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
            idx.data_ptr<int>(), hard_mask.data_ptr<int>(),
            depth.data_ptr<float>(), reinterpret_cast<float2*>(uv.data_ptr<float>()), alpha.data_ptr<float>(),
            radius, sigma
        );
    } else if (N_CANDIDATES <= 8) {
        cast_radius_rays_kernel<8><<<(N_RAYS + threads - 1) / threads, threads>>>(
            N_RAYS, N_CANDIDATES,
            reinterpret_cast<float3*>(ray_ori.data_ptr<float>()),
            reinterpret_cast<float3*>(ray_dir.data_ptr<float>()),
            reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
            reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
            reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
            tree.orig_idx.data_ptr<int64_t>(),
            reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
            idx.data_ptr<int>(), hard_mask.data_ptr<int>(),
            depth.data_ptr<float>(), reinterpret_cast<float2*>(uv.data_ptr<float>()), alpha.data_ptr<float>(),
            radius, sigma
        );
    } else if (N_CANDIDATES <= 16) {
        cast_radius_rays_kernel<16><<<(N_RAYS + threads - 1) / threads, threads>>>(
            N_RAYS, N_CANDIDATES,
            reinterpret_cast<float3*>(ray_ori.data_ptr<float>()),
            reinterpret_cast<float3*>(ray_dir.data_ptr<float>()),
            reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
            reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
            reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
            tree.orig_idx.data_ptr<int64_t>(),
            reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
            idx.data_ptr<int>(), hard_mask.data_ptr<int>(),
            depth.data_ptr<float>(), reinterpret_cast<float2*>(uv.data_ptr<float>()), alpha.data_ptr<float>(),
            radius, sigma
        );
    } else {
        cast_radius_rays_kernel<32><<<(N_RAYS + threads - 1) / threads, threads>>>(
            N_RAYS, N_CANDIDATES,
            reinterpret_cast<float3*>(ray_ori.data_ptr<float>()),
            reinterpret_cast<float3*>(ray_dir.data_ptr<float>()),
            reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
            reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
            reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
            tree.orig_idx.data_ptr<int64_t>(),
            reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
            idx.data_ptr<int>(), hard_mask.data_ptr<int>(),
            depth.data_ptr<float>(), reinterpret_cast<float2*>(uv.data_ptr<float>()), alpha.data_ptr<float>(),
            radius, sigma
        );
    }

    // Unsort indices to match original ray order
    torch::Tensor ray_inverse = torch::empty_like(ray_order);
    ray_inverse.index_put_({ray_order}, torch::arange(0, ray_order.size(0), torch::TensorOptions().dtype(ray_order.dtype()).device(ray_order.device())));

    return {
        idx.index({torch::indexing::Slice(), ray_inverse}),
        depth.index({torch::indexing::Slice(), ray_inverse}),
        uv.index({torch::indexing::Slice(), ray_inverse}),
        alpha.index({torch::indexing::Slice(), ray_inverse}),
        hard_mask.index({ray_inverse})
    };
}


__global__ void cast_radius_rays_bwd_kernel(
    const int N_RAYS,
    const float3* __restrict__ ray_ori,
    const float3* __restrict__ ray_dir,
    const int N_CANDIDATES,
    const int* __restrict__ tri_idx,
    const int* __restrict__ hard_mask,
    const float3* __restrict__ tri,
    const float* __restrict__ grad_depth,
    const float* __restrict__ grad_uv,
    const float* __restrict__ grad_alpha,
    float* __restrict__ grad_tri,
    const float sigma
) {
    const int gidx = global_idx();
    if (gidx >= N_RAYS * N_CANDIDATES) return;

    const int r_idx = gidx % N_RAYS;
    // const int k_idx = gidx / N_RAYS;

    const int found_idx = tri_idx[gidx];
    if (found_idx < 0) return;

    // read ray and triangle
    const float3 ray_o = ray_ori[r_idx];
    const float3 ray_d_orig = ray_dir[r_idx];
    const float3 ray_d = normalize(ray_d_orig);

    const float3* curr_tri = tri + found_idx * 3;
    const float3 v0 = curr_tri[0], v1 = curr_tri[1], v2 = curr_tri[2];

    float3 v0_grad = make_float3(0.0f, 0.0f, 0.0f);
    float3 v1_grad = make_float3(0.0f, 0.0f, 0.0f);
    float3 v2_grad = make_float3(0.0f, 0.0f, 0.0f);

    if (hard_mask[r_idx] == found_idx) {
        // Hard intersection logic.
        const float ax = fabsf(ray_d.x), ay = fabsf(ray_d.y), az = fabsf(ray_d.z);
        const int kz = (az > ay) ? (az > ax ? 2 : 0) : (ay > ax ? 1 : 0);
        int kx, ky;
        if ((&ray_d.x)[kz] > 0.0f){
            kx = (kz + 1) % 3; ky = (kz + 2) % 3;
        } else {
            kx = (kz + 2) % 3; ky = (kz + 1) % 3;
        }

        const float Sz = 1.0f / (&ray_d.x)[kz];
        const float Sx = (&ray_d.x)[kx] * Sz;
        const float Sy = (&ray_d.x)[ky] * Sz;

        const float3 A = v0 - ray_o, B = v1 - ray_o, C = v2 - ray_o;

        const float Ax = (&A.x)[kx] - Sx * (&A.x)[kz];
        const float Ay = (&A.x)[ky] - Sy * (&A.x)[kz];
        const float Bx = (&B.x)[kx] - Sx * (&B.x)[kz];
        const float By = (&B.x)[ky] - Sy * (&B.x)[kz];
        const float Cx = (&C.x)[kx] - Sx * (&C.x)[kz];
        const float Cy = (&C.x)[ky] - Sy * (&C.x)[kz];

        const double U = static_cast<double>(Cx) * By - static_cast<double>(Cy) * Bx;
        const double V = static_cast<double>(Ax) * Cy - static_cast<double>(Ay) * Cx;
        const double W = static_cast<double>(Bx) * Ay - static_cast<double>(By) * Ax;
        const double det = U + V + W;
        const double invdet = 1.0 / det;

        const float Az = (&A.x)[kz] * Sz;
        const float Bz = (&B.x)[kz] * Sz;
        const float Cz = (&C.x)[kz] * Sz;

        const float dist = static_cast<float>((U * Az + V * Bz + W * Cz) * invdet);
        const float u = static_cast<float>(V * invdet);
        const float v = static_cast<float>(W * invdet);

        const float du = grad_uv[gidx * 2 + 0];
        const float dv = grad_uv[gidx * 2 + 1];
        const float ddist = grad_depth[gidx] / length(ray_d_orig);

        // ----- Reverse pass -----
        double gU = 0.0, gV = 0.0, gW = 0.0;
        double gDet = 0.0, gInvDet = 0.0;
        float gAx = 0.0f, gAy = 0.0f, gBx = 0.0f, gBy = 0.0f, gCx = 0.0f, gCy = 0.0f;
        float gAz = 0.0f, gBz = 0.0f, gCz = 0.0f;
        float gA[3] = {0.0f, 0.0f, 0.0f};
        float gB[3] = {0.0f, 0.0f, 0.0f};
        float gC[3] = {0.0f, 0.0f, 0.0f};

        // dist contribution
        if (dist > 0.0f){
            gInvDet += ddist * (U * Az + V * Bz + W * Cz);

            const double factor = ddist * invdet;
            gU += factor * Az;
            gAz += static_cast<float>(factor * U);
            gV += factor * Bz;
            gBz += static_cast<float>(factor * V);
            gW += factor * Cz;
            gCz += static_cast<float>(factor * W);
        }
        // u contribution
        if (u > 0.0f && u < 1.0f){
            gV += du * invdet;
            gInvDet += du * V;
        }

        // v contribution
        if (v > 0.0f && v < 1.0f){
            gW += dv * invdet;
            gInvDet += dv * W;
        }

        // invdet = 1 / det
        gDet += -1.0 * gInvDet * invdet * invdet;

        // det = U + V + W
        gU += gDet;
        gV += gDet;
        gW += gDet;

        // U = Cx*By - Cy*Bx
        gCx += static_cast<float>(gU) * By;
        gBy += static_cast<float>(gU) * Cx;
        gCy += static_cast<float>(gU) * (-Bx);
        gBx += static_cast<float>(gU) * (-Cy);

        // V = Ax*Cy - Ay*Cx
        gAx += static_cast<float>(gV) * Cy;
        gCy += static_cast<float>(gV) * Ax;
        gAy += static_cast<float>(gV) * (-Cx);
        gCx += static_cast<float>(gV) * (-Ay);

        // W = Bx*Ay - By*Ax
        gBx += static_cast<float>(gW) * Ay;
        gAy += static_cast<float>(gW) * Bx;
        gBy += static_cast<float>(gW) * (-Ax);
        gAx += static_cast<float>(gW) * (-By);

        // Ax = A_kx - Sx * A_kz
        gA[kx] += gAx;
        gA[kz] += gAx * (-Sx);

        // Ay = A_ky - Sy * A_kz
        gA[ky] += gAy;
        gA[kz] += gAy * (-Sy);

        // Bx, By
        gB[kx] += gBx;
        gB[kz] += gBx * (-Sx);
        gB[ky] += gBy;
        gB[kz] += gBy * (-Sy);

        // Cx, Cy
        gC[kx] += gCx;
        gC[kz] += gCx * (-Sx);
        gC[ky] += gCy;
        gC[kz] += gCy * (-Sy);

        // Az = A_kz * Sz, etc. (Sz treated constant w.r.t vertices)
        gA[kz] += gAz * Sz;
        gB[kz] += gBz * Sz;
        gC[kz] += gCz * Sz;

        v0_grad += make_float3(gA[0], gA[1], gA[2]);
        v1_grad += make_float3(gB[0], gB[1], gB[2]);
        v2_grad += make_float3(gC[0], gC[1], gC[2]);

    } else {
        // Radius intersection logic
        const float3 w = ray_d;
        float3 up = make_float3(0.0f, 1.0f, 0.0f);
        if (fabsf(w.y) > 0.99f) up = make_float3(1.0f, 0.0f, 0.0f);
        const float3 u_axis = normalize(cross(w, up));
        const float3 v_axis = normalize(cross(w, u_axis));

        const float3 p0 = v0 - ray_o;
        const float3 p1 = v1 - ray_o;
        const float3 p2 = v2 - ray_o;

        const float2 p0xy = make_float2(dot(p0, u_axis), dot(p0, v_axis));
        const float2 p1xy = make_float2(dot(p1, u_axis), dot(p1, v_axis));
        const float2 p2xy = make_float2(dot(p2, u_axis), dot(p2, v_axis));

        const float2 origin = make_float2(0.0f, 0.0f);
        const double w0 = cross2d(p1xy - origin, p2xy - origin);
        const double w1 = cross2d(p2xy - origin, p0xy - origin);
        const double w2 = cross2d(p0xy - origin, p1xy - origin);

        const bool inside = (w0 > 0.0 && w1 > 0.0 && w2 > 0.0) || (w0 < 0.0 && w1 < 0.0 && w2 < 0.0);

        float t01, t12, t20;
        const float d01 = point_to_segment_dist_sq_2d(origin, p0xy, p1xy, t01);
        const float d12 = point_to_segment_dist_sq_2d(origin, p1xy, p2xy, t12);
        const float d20 = point_to_segment_dist_sq_2d(origin, p2xy, p0xy, t20);

        float b0, b1, b2;
        float d_val;


        // Logic for alpha gradient
        // alpha = exp(-max(d, 0)/sigma)
        // d_alpha = alpha * (-1/sigma) * d(max(d,0))
        // if d > 0: d_alpha = alpha * -1/sigma * d(d)
        // if d <= 0: d_alpha = 0
        if (inside) {
            d_val = -fminf(d01, fminf(d12, d20));
//             const float alpha_val = 1.0f;
            // grad_d remains 0 as alpha is constant 1 here

            double invarea = 1.0 / (w0 + w1 + w2 + 1e-12);
            b0 = fminf(fmaxf(static_cast<float>(w0 * invarea), 0.0f), 1.0f);
            b1 = fminf(fmaxf(static_cast<float>(w1 * invarea), 0.0f), 1.0f);
            b2 = fminf(fmaxf(static_cast<float>(w2 * invarea), 0.0f), 1.0f);

            // Gradients for barycentrics
            // b0 = w0 * invarea, etc.
            // db0 = dw0 * invarea + w0 * dinvarea
            // dinvarea = -1/area^2 * (dw0+dw1+dw2) = -invarea^2 * (dw0+dw1+dw2)

            float d_depth = grad_depth[gidx] / length(ray_d_orig);
            float d_b1 = grad_uv[gidx * 2 + 0];
            float d_b2 = grad_uv[gidx * 2 + 1];

            // Propagate depth gradient to b and v
            // depth = dot(q - ray_o, w) = dot(b0 v0 + b1 v1 + b2 v2 - ray_o, w)
            // d_depth/dv0 = b0 * w, etc.
            v0_grad += d_depth * b0 * w;
            v1_grad += d_depth * b1 * w;
            v2_grad += d_depth * b2 * w;

            // d_depth/db0 = dot(v0, w), etc.
            float d_depth_b0 = dot(v0, w) * d_depth;
            float d_depth_b1 = dot(v1, w) * d_depth;
            float d_depth_b2 = dot(v2, w) * d_depth;

            float total_db0 = d_depth_b0;
            float total_db1 = d_depth_b1 + d_b1;
            float total_db2 = d_depth_b2 + d_b2;

            double dinvarea = -(total_db0 * w0 + total_db1 * w1 + total_db2 * w2) * invarea * invarea;
            double dw0 = total_db0 * invarea + dinvarea;
            double dw1 = total_db1 * invarea + dinvarea;
            double dw2 = total_db2 * invarea + dinvarea;

            // dw0 / dp1xy = (p2xy.y, -p2xy.x) (rotated)
            // w0 = p1x p2y - p1y p2x
            // dw0/dp1x = p2y, dw0/dp1y = -p2x
            // dw0/dp2x = -p1y, dw0/dp2y = p1x

            float2 dp1xy = make_float2(static_cast<float>(dw0 * p2xy.y - dw2 * p0xy.y), static_cast<float>(-dw0 * p2xy.x + dw2 * p0xy.x));
            float2 dp2xy = make_float2(static_cast<float>(dw1 * p0xy.y - dw0 * p1xy.y), static_cast<float>(-dw1 * p0xy.x + dw0 * p1xy.x));
            float2 dp0xy = make_float2(static_cast<float>(dw2 * p1xy.y - dw1 * p2xy.y), static_cast<float>(-dw2 * p1xy.x + dw1 * p2xy.x));

            // Back to 3D
            v0_grad += dp0xy.x * u_axis + dp0xy.y * v_axis;
            v1_grad += dp1xy.x * u_axis + dp1xy.y * v_axis;
            v2_grad += dp2xy.x * u_axis + dp2xy.y * v_axis;

        } else {
            d_val = d01;
            int min_idx = 0;
            if (d12 < d_val) { d_val = d12; min_idx = 1; }
            if (d20 < d_val) { d_val = d20; min_idx = 2; }

            // alpha_val = expf(-d_val / sigma);
//             const float alpha_val = 1.0f - fminf(fmaxf(d_val, 0.0f) / sigma, 1.0f);

            // Gradient from alpha
            // dL/dd_val
            // alpha = exp(-d/sigma)
            // d_alpha/dd = -1/sigma * alpha
            // float dL_dd = grad_alpha[gidx] * alpha_val * (-1.0f / sigma);

            // Linear: d_alpha/dd = -1/sigma
            float dL_dd = grad_alpha[gidx] * (-1.0f / sigma);
            if (d_val < 0.0f || d_val > sigma) dL_dd = 0.0f;

            // Gradients from uv and depth
            float d_depth = grad_depth[gidx] / length(ray_d_orig);
            float d_b1 = grad_uv[gidx * 2 + 0];
            float d_b2 = grad_uv[gidx * 2 + 1];

            // Calculate b values and their gradients w.r.t t
            float t_val;

            // We also need dL/dt
            float dL_dt = 0.0f;

            // Also need direct depth derivatives
            // q = b0 v0 + b1 v1 + b2 v2

            if (min_idx == 0) { // Edge 01
                t_val = t01;
                b0 = 1.0f - t_val; b1 = t_val; b2 = 0.0f;
                // db0/dt = -1, db1/dt = 1

                // depth = dot((1-t)v0 + t v1 - ray, w)
                // d_depth/dt = dot(v1 - v0, w)
                dL_dt += d_depth * dot(v1 - v0, w);

                // b1 = t -> dL/dt += dL/db1 = d_b1
                dL_dt += d_b1;
                // b2 = 0

                // Direct v0, v1 gradients from depth (holding t constant)
                v0_grad += d_depth * (1.0f - t_val) * w;
                v1_grad += d_depth * t_val * w;

                // Gradient of d01 w.r.t p0xy, p1xy
                // d = sqrt(len2) where len2 = |p - (a + t ba)|^2
                // But simpler: d_val comes from point_to_segment
                // Let's compute grads of d_val and t_val w.r.t p0xy, p1xy

                // Re-run point_to_segment logic for gradients
                float2 a = p0xy, b = p1xy, p = origin;
                float2 ba = b - a;
                float2 pa = p - a;
                double ba_len2 = static_cast<double>(ba.x) * ba.x + static_cast<double>(ba.y) * ba.y;
                double dot_pa_ba = static_cast<double>(pa.x) * ba.x + static_cast<double>(pa.y) * ba.y;

                // t unclamped
                float t_raw = static_cast<float>(dot_pa_ba / (ba_len2 + 1e-12));
                float t_final = fminf(fmaxf(t_raw, 0.0f), 1.0f);

                float2 q = a + ba * t_final;
                float2 pq = p - q;
                float d_res = static_cast<float>(sqrt(static_cast<double>(pq.x) * pq.x + static_cast<double>(pq.y) * pq.y + 1e-12));

                // Derivatives
                // dL_total / d_something
                // We have dL/dd (dL_dd) and dL/dt (dL_dt)

                // dL/dq
                // d = |p-q| => dd/dq = -(p-q)/d = q/d (since p=0)
                float2 dd_dq = q / (d_res + 1e-12f);
                float2 dL_dq = dL_dd * dd_dq;

                // q = a + ba * t
                // dq/da = 1 - t, dq/db = t, dq/dt = ba

                float2 dL_da = dL_dq * (1.0f - t_final);
                float2 dL_db = dL_dq * t_final;
                float dL_dt_from_q = dot(dL_dq, ba);
                dL_dt += dL_dt_from_q;

                // Now we need dt/da and dt/db if t is not clamped
                if (t_raw > 0.0f && t_raw < 1.0f) {
                     // t = dot(pa, ba) / dot(ba, ba) = (px-ax)(bx-ax) + ... / |b-a|^2
                     // t = (-ax(bx-ax) - ay(by-ay)) / L2
                     // This is getting messy.
                     // Alternative: Autodiff t calculation
                     // t = (p-a).dot(b-a) / (b-a)^2
                     // let N = (p-a).dot(b-a), D = (b-a).dot(b-a)
                     // dt/da = (dN/da * D - N * dD/da) / D^2
                     // N = -a.(b-a) (p=0) = -a.b + a.a
                     // D = b.b - 2a.b + a.a
                     // dN/da = -b + 2a
                     // dD/da = -2b + 2a

                     // dt/db
                     // N = -a.b + a.a
                     // dN/db = -a
                     // dD/db = 2b - 2a

                     float2 dN_da = -b + 2.0f * a;
                     float2 dD_da = -2.0f * b + 2.0f * a;
                     float2 dt_da = make_float2(static_cast<float>((dN_da.x - dot_pa_ba / ba_len2 * dD_da.x) / (ba_len2 + 1e-12)), static_cast<float>((dN_da.y - dot_pa_ba / ba_len2 * dD_da.y) / (ba_len2 + 1e-12)));

                     float2 dN_db = -a;
                     float2 dD_db = 2.0f * b - 2.0f * a;
                     float2 dt_db = make_float2(static_cast<float>((dN_db.x - dot_pa_ba / ba_len2 * dD_db.x) / (ba_len2 + 1e-12)), static_cast<float>((dN_db.y - dot_pa_ba / ba_len2 * dD_db.y) / (ba_len2 + 1e-12)));

                     dL_da += dL_dt * dt_da;
                     dL_db += dL_dt * dt_db;
                }

                // dL_da corresponds to grad_p0xy, dL_db to grad_p1xy
                v0_grad += dL_da.x * u_axis + dL_da.y * v_axis;
                v1_grad += dL_db.x * u_axis + dL_db.y * v_axis;

            } else if (min_idx == 1) { // Edge 12
                // p1xy, p2xy
                float2 a = p1xy, b = p2xy, p = origin;
                // ... Copy paste logic with index shift ...
                // a->p1, b->p2
                t_val = t12;
                b0 = 0.0f; b1 = 1.0f - t_val; b2 = t_val;

                dL_dt += d_depth * dot(v2 - v1, w);
                dL_dt += -d_b1 + d_b2;

                v1_grad += d_depth * (1.0f - t_val) * w;
                v2_grad += d_depth * t_val * w;

                float2 ba = b - a;
                float2 pa = p - a;
                double ba_len2 = static_cast<double>(ba.x) * ba.x + static_cast<double>(ba.y) * ba.y;
                double dot_pa_ba = static_cast<double>(pa.x) * ba.x + static_cast<double>(pa.y) * ba.y;

                float t_raw = static_cast<float>(dot_pa_ba / (ba_len2 + 1e-12));
                float t_final = fminf(fmaxf(t_raw, 0.0f), 1.0f);

                float2 q = a + ba * t_final;
                float2 pq = p - q;
                float d_res = static_cast<float>(sqrt(static_cast<double>(pq.x) * pq.x + static_cast<double>(pq.y) * pq.y + 1e-12));

                float2 dd_dq = q / (d_res + 1e-12f);
                float2 dL_dq = dL_dd * dd_dq;

                float2 dL_da = dL_dq * (1.0f - t_final);
                float2 dL_db = dL_dq * t_final;
                float dL_dt_from_q = dot(dL_dq, ba);
                dL_dt += dL_dt_from_q;

                if (t_raw > 0.0f && t_raw < 1.0f) {
                     float2 dN_da = -b + 2.0f * a;
                     float2 dD_da = -2.0f * b + 2.0f * a;
                     float2 dt_da = make_float2(static_cast<float>((dN_da.x - dot_pa_ba / ba_len2 * dD_da.x) / (ba_len2 + 1e-12)), static_cast<float>((dN_da.y - dot_pa_ba / ba_len2 * dD_da.y) / (ba_len2 + 1e-12)));;

                     float2 dN_db = -a;
                     float2 dD_db = 2.0f * b - 2.0f * a;
                     float2 dt_db = make_float2(static_cast<float>((dN_db.x - dot_pa_ba / ba_len2 * dD_db.x) / (ba_len2 + 1e-12)), static_cast<float>((dN_db.y - dot_pa_ba / ba_len2 * dD_db.y) / (ba_len2 + 1e-12)));

                     dL_da += dL_dt * dt_da;
                     dL_db += dL_dt * dt_db;
                }

                v1_grad += dL_da.x * u_axis + dL_da.y * v_axis;
                v2_grad += dL_db.x * u_axis + dL_db.y * v_axis;

            } else { // Edge 20
                // p2xy, p0xy
                float2 a = p2xy, b = p0xy, p = origin;
                t_val = t20;
                b0 = t_val; b1 = 0.0f; b2 = 1.0f - t_val;

                // dL_dt = d_depth * dot(v0 - v2, w) - d_b2;
                dL_dt += d_depth * dot(v0 - v2, w);
                dL_dt += -d_b2;

                v2_grad += d_depth * (1.0f - t_val) * w;
                v0_grad += d_depth * t_val * w;

                float2 ba = b - a;
                float2 pa = p - a;
                double ba_len2 = static_cast<double>(ba.x) * ba.x + static_cast<double>(ba.y) * ba.y;
                double dot_pa_ba = static_cast<double>(pa.x) * ba.x + static_cast<double>(pa.y) * ba.y;

                float t_raw = static_cast<float>(dot_pa_ba / (ba_len2 + 1e-12));
                float t_final = fminf(fmaxf(t_raw, 0.0f), 1.0f);

                float2 q = a + ba * t_final;
                float2 pq = p - q;
                float d_res = static_cast<float>(sqrt(static_cast<double>(pq.x) * pq.x + static_cast<double>(pq.y) * pq.y + 1e-12));

                float2 dd_dq = q / (d_res + 1e-12f);
                float2 dL_dq = dL_dd * dd_dq;

                float2 dL_da = dL_dq * (1.0f - t_final);
                float2 dL_db = dL_dq * t_final;
                float dL_dt_from_q = dot(dL_dq, ba);
                dL_dt += dL_dt_from_q;

                if (t_raw > 0.0f && t_raw < 1.0f) {
                     float2 dN_da = -b + 2.0f * a;
                     float2 dD_da = -2.0f * b + 2.0f * a;
                     float2 dt_da = make_float2(static_cast<float>((dN_da.x - dot_pa_ba / ba_len2 * dD_da.x) / (ba_len2 + 1e-12)), static_cast<float>((dN_da.y - dot_pa_ba / ba_len2 * dD_da.y) / (ba_len2 + 1e-12)));

                     float2 dN_db = -a;
                     float2 dD_db = 2.0f * b - 2.0f * a;
                     float2 dt_db = make_float2(static_cast<float>((dN_db.x - dot_pa_ba / ba_len2 * dD_db.x) / (ba_len2 + 1e-12)), static_cast<float>((dN_db.y - dot_pa_ba / ba_len2 * dD_db.y) / (ba_len2 + 1e-12)));

                     dL_da += dL_dt * dt_da;
                     dL_db += dL_dt * dt_db;
                }

                v2_grad += dL_da.x * u_axis + dL_da.y * v_axis;
                v0_grad += dL_db.x * u_axis + dL_db.y * v_axis;
            }
        }
    }

    float* curr_grad_tri = grad_tri + 9 * found_idx;
    atomicAdd(curr_grad_tri + 3 * 0 + 0, v0_grad.x); atomicAdd(curr_grad_tri + 3 * 0 + 1, v0_grad.y); atomicAdd(curr_grad_tri + 3 * 0 + 2, v0_grad.z);
    atomicAdd(curr_grad_tri + 3 * 1 + 0, v1_grad.x); atomicAdd(curr_grad_tri + 3 * 1 + 1, v1_grad.y); atomicAdd(curr_grad_tri + 3 * 1 + 2, v1_grad.z);
    atomicAdd(curr_grad_tri + 3 * 2 + 0, v2_grad.x); atomicAdd(curr_grad_tri + 3 * 2 + 1, v2_grad.y); atomicAdd(curr_grad_tri + 3 * 2 + 2, v2_grad.z);

}



torch::Tensor cuda_cast_radius_rays_bwd(
    torch::Tensor tri,
    torch::Tensor ray_ori,
    torch::Tensor ray_dir,
    torch::Tensor tri_idx,
    torch::Tensor hard_mask,
    torch::Tensor grad_depth,
    torch::Tensor grad_uv,
    torch::Tensor grad_alpha,
    float sigma
) {
    constexpr int threads = 128;
    auto device = tri.device();

    const int N_TRI = tri.size(0), N_RAYS = ray_ori.size(0), N_CANDIDATES = tri_idx.size(0);
    auto grad_tri = torch::zeros({N_TRI, 3, 3}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

    cast_radius_rays_bwd_kernel<<<(N_RAYS * N_CANDIDATES + threads - 1) / threads, threads>>>(
        N_RAYS,
        reinterpret_cast<float3*>(ray_ori.data_ptr<float>()),
        reinterpret_cast<float3*>(ray_dir.data_ptr<float>()),
        N_CANDIDATES,
        tri_idx.data_ptr<int>(),
        hard_mask.data_ptr<int>(),
        reinterpret_cast<float3*>(tri.data_ptr<float>()),
        grad_depth.data_ptr<float>(),
        grad_uv.data_ptr<float>(),
        grad_alpha.data_ptr<float>(),
        grad_tri.data_ptr<float>(),
        sigma
    );

    return grad_tri;
}
