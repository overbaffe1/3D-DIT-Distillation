

#include <mesh.h>
#include <bvh.h>
#include <cutil_math.h>

#define INT_EPS 1e-12 // low threshold to avoid false positives inside mesh
#define EXT_EPS 1e-6


__device__ __forceinline__ float p_seg_dist(const float3 &p, const float3 &v0, const float3 &v1, float &t){
   const float3 dx = v1 - v0;
   t = fminf(fmaxf(static_cast<float>(((v1.x - p.x) * static_cast<double>(dx.x) + (v1.y - p.y) * static_cast<double>(dx.y) + (v1.z - p.z) * static_cast<double>(dx.z)) / (static_cast<double>(dx.x) * dx.x + static_cast<double>(dx.y) * dx.y + static_cast<double>(dx.z) * dx.z)), 0.0f), 1.0f);
   const float3 dp = p - (v1 + t * (v0 - v1));
   return static_cast<float>(sqrt(static_cast<double>(dp.x) * dp.x + static_cast<double>(dp.y) * dp.y + static_cast<double>(dp.z) * dp.z));
}

__device__  __forceinline__ float p_tri_dist(
    const float3 &p, const float3 &v0, const float3 &v1, const float3 &v2,
    float &u, float &v
){

    const float3 x02 = v0 - v2, x12 = v1 - v2, xp2 = p - v2;
    const double m02 = (static_cast<double>(x02.x) * x02.x + static_cast<double>(x02.y) * x02.y + static_cast<double>(x02.z) * x02.z);
    const double m12 = (static_cast<double>(x12.x) * x12.x + static_cast<double>(x12.y) * x12.y + static_cast<double>(x12.z) * x12.z);
    const double d = (static_cast<double>(x02.x) * x12.x + static_cast<double>(x02.y) * x12.y + static_cast<double>(x02.z) * x12.z);
    const float a = dot(x02, xp2), b = dot(x12, xp2);
    const double invdet = 1.0 / fmax(m02 * m12 - d * d, 1e-30);

    const float w12 = static_cast<float>(invdet * (m12 * a - d * b));
    const float w20 = static_cast<float>(invdet * (m02 * b - d * a));
    const float w01 = 1.0f - w12 - w20;

    if (w01 >= 0.0f && w12 >= 0.0f && w20 >= 0.0f){
        u = fminf(fmaxf(static_cast<float>(w20), 0.0f), 1.0f); 
        v = fminf(fmaxf(static_cast<float>(w01), 0.0f), 1.0f);  
        const float w = 1.0f - u - v;
        const float3 pp = (w * v0 + u * v1 + v * v2);
        return length(p - pp);
    }

    if (w12 > 0.0f){
        float t1, t2;
        const float dist1 = p_seg_dist(p, v0, v1, t1);
        const float dist2 = p_seg_dist(p, v0, v2, t2);
        if (dist1 < dist2) {
            // v0 (1 - u - v) + u v1 + v v2 = t1 v0 + (1 - t1) v1
            u = static_cast<float>(1.0f - t1);
            v = 0.0f;
            return dist1;
        } else {
            // v0 (1 - u - v) + u v1 + v v2 = t2 v0 + (1 - t2) v2
            u = 0.0f;
            v = static_cast<float>(1.0f - t2);
            return dist2;
        }
    }

    if (w20 > 0.0f){
        float t1, t2;
        const float dist1 = p_seg_dist(p, v0, v1, t1);
        const float dist2 = p_seg_dist(p, v1, v2, t2);
        if (dist1 < dist2) {
            u = static_cast<float>(1.0f - t1);
            v = 0.0f;
            return dist1;
        } else {
            // v0 (1 - u - v) + u v1 + v v2 = t2 v1 + (1 - t2) v2
            u = static_cast<float>(t2);
            v = static_cast<float>(1.0f - t2);
            return dist2;
        }
    }

    // w01
    {
        float t1, t2;
        const float dist1 = p_seg_dist(p, v0, v2, t1);
        const float dist2 = p_seg_dist(p, v1, v2, t2);
        if (dist1 < dist2) {
            // v0 (1 - u - v) + u v1 + v v2 = t1 v0 + (1 - t1) v2
            u = 0.0f;
            v = static_cast<float>(1.0f - t1);
            return dist1;
        } else {
            // v0 (1 - u - v) + u v1 + v v2 = t2 v1 + (1 - t2) v2
            u = static_cast<float>(t2);
            v = static_cast<float>(1.0f - t2);
            return dist2;
        }
    }

}

__device__  bool seg_tri_intersection(const float3 &p1, const float3 &p2, const float3 &v0, const float3 &v1, const float3 &v2, const float shift){

    const float3 v00 = v0 + normalize(v0 - (v1 + v2) / 2.0f) * shift;
    const float3 v11 = v1 + normalize(v1 - (v0 + v2) / 2.0f) * shift;
    const float3 v22 = v2 + normalize(v2 - (v0 + v1) / 2.0f) * shift;

    const float3 edge0 = v11 - v00;
    const float3 edge1 = v22 - v00;

    float3 pvec = cross(p2 - p1, edge1);
    float det = dot(edge0, pvec);
    if (fabs(det) < INT_EPS) return false;

    float3 tvec = p1 - v00;
    float3 qvec = cross(tvec, edge0);

    float u = dot(tvec, pvec) / det;
    if (u < 0.0f - EXT_EPS || u > 1.0f + EXT_EPS) return false;

    float v = dot(p2 - p1, qvec) / det;
    if (v < 0.0f - EXT_EPS || u + v > 1.0f + EXT_EPS) return false;

    float t = dot(edge1, qvec) / det;
    if(t < 0.0f - EXT_EPS || t > 1.0f + EXT_EPS) return false;

    return true;
}



__device__ bool ray_tri_intersection(
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


__device__  bool __bbox_intersect(const float3 &p, const float3 &d, const float3 &bb_min_orig, const float3 &bb_max_orig, float ext, float max_dist){
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

__device__ bool ray_bbox_intersect(const float3 &p, const float3 &d, const float3 &bb_min_orig, const float3 &bb_max_orig){
    return __bbox_intersect(p, d, bb_min_orig, bb_max_orig, FLT_EPSILON, 1e9);
}

__device__ bool seg_bbox_intersect(const float3 &p, const float3 &d, const float3 &bb_min_orig, const float3 &bb_max_orig, float ext){
    return __bbox_intersect(p, d, bb_min_orig, bb_max_orig, ext, 1.0f);
}

__global__ void brute_sign_kernel(
    const float* q,
    const int n,
    const float* tri,
    const int n_tri,
    float *opacity
){

    const int thread_idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= n)
        return;

    const float* curr_q = q + 3 * thread_idx;
    const float3 ray_o = make_float3(curr_q[0], curr_q[1], curr_q[2]);
    const float3 ray_d = make_float3(0.0f, 0.0f, 1.0f);

    unsigned int intersections = 0;
    for(unsigned int tri_idx = 0; tri_idx < n_tri; tri_idx++){

        const float* curr_triangle = tri + 9 * tri_idx;
        const float3 v0 = make_float3(curr_triangle[3 * 0 + 0], curr_triangle[3 * 0 + 1], curr_triangle[3 * 0 + 2]);
        const float3 v1 = make_float3(curr_triangle[3 * 1 + 0], curr_triangle[3 * 1 + 1], curr_triangle[3 * 1 + 2]);
        const float3 v2 = make_float3(curr_triangle[3 * 2 + 0], curr_triangle[3 * 2 + 1], curr_triangle[3 * 2 + 2]);

        float dist, u, v;
        intersections += (unsigned int) ray_tri_intersection(ray_o, ray_d, v0, v1, v2, dist,  u, v);
    }

    const float opacity_value = (float) ((intersections + 1) % 2);
    opacity[thread_idx] = opacity_value * 2 - 1;
}


__global__ void brute_dist_kernel(
    const float* q,
    const int n,
    const float* tri,
    const int n_tri,
    float *dist
){

    const int thread_idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= n)
        return;

    const float* curr_q = q + 3 * thread_idx;
    const float3 p = make_float3(curr_q[0], curr_q[1], curr_q[2]);

    float min_dist = 1e6;
    for(unsigned int tri_idx = 0; tri_idx < n_tri; tri_idx++){

        const float* curr_triangle = tri + 9 * tri_idx;
        const float3 v0 = make_float3(curr_triangle[3 * 0 + 0], curr_triangle[3 * 0 + 1], curr_triangle[3 * 0 + 2]);
        const float3 v1 = make_float3(curr_triangle[3 * 1 + 0], curr_triangle[3 * 1 + 1], curr_triangle[3 * 1 + 2]);
        const float3 v2 = make_float3(curr_triangle[3 * 2 + 0], curr_triangle[3 * 2 + 1], curr_triangle[3 * 2 + 2]);

        float tri_u, tri_v;
        const float d = p_tri_dist(p, v0, v1, v2, tri_u, tri_v);
        min_dist = fminf(min_dist, d);
    }

    dist[thread_idx] = min_dist;
}


__global__ void brute_sdf_kernel(
    const float* q,
    const int n,
    const float* tri,
    const int n_tri,
    float *sdf
){

    const int thread_idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= n)
        return;

    const float* curr_q = q + 3 * thread_idx;
    const float3 ray_o = make_float3(curr_q[0], curr_q[1], curr_q[2]);
    const float3 ray_d = make_float3(0.0f, 0.0f, 1.0f);

    unsigned int intersections = 0;
    float min_dist = 1e6;
    for(unsigned int tri_idx = 0; tri_idx < n_tri; tri_idx++){

        const float* curr_triangle = tri + 9 * tri_idx;
        const float3 v0 = make_float3(curr_triangle[3 * 0 + 0], curr_triangle[3 * 0 + 1], curr_triangle[3 * 0 + 2]);
        const float3 v1 = make_float3(curr_triangle[3 * 1 + 0], curr_triangle[3 * 1 + 1], curr_triangle[3 * 1 + 2]);
        const float3 v2 = make_float3(curr_triangle[3 * 2 + 0], curr_triangle[3 * 2 + 1], curr_triangle[3 * 2 + 2]);

        // FIND MIN DISTANCE
        float tri_u, tri_v;
        const float d = p_tri_dist(ray_o, v0, v1, v2, tri_u, tri_v);
        min_dist = fminf(min_dist, d);

        // CHECK INTERSECTION
        float tri_depth;
        intersections += (unsigned int) ray_tri_intersection(ray_o, ray_d, v0, v1, v2, tri_depth, tri_u, tri_v);
    }

    const float opacity = (float) ((intersections + 1) % 2);
    sdf[thread_idx] = (opacity * 2 - 1) * min_dist;
}

void cuda_sign(torch::Tensor q, torch::Tensor tri, torch::Tensor o){

    const int n = q.size(0);
    const int threads = 512;

    brute_sign_kernel<<<(n + threads - 1) / threads, threads>>>(
        q.data_ptr<float>(),
        n,
        tri.data_ptr<float>(),
        (int) tri.size(0),
        o.data_ptr<float>()
    );
}

void cuda_dist(torch::Tensor q, torch::Tensor tri, torch::Tensor d) {
    const int n = q.size(0);
    const int threads = 512;

    brute_dist_kernel<<<(n + threads - 1) / threads, threads>>>(
        q.data_ptr<float>(),
        n,
        tri.data_ptr<float>(),
        (int) tri.size(0),
        d.data_ptr<float>()
    );
}


#define MAX_DEPTH 32
#define STACK_SIZE (MAX_DEPTH * 2)

__device__ float pdist(
    const float3 p,
    const int3* __restrict__ nodes, const int2* __restrict__ ranges,
    const float3* __restrict__ bbox, const float3* __restrict__ tri,
    int* min_idx, float2* min_uv, float3* min_pcd
){

    float mindist = 1e6f;
    int found_idx = -1; float found_u = 0.0, found_v = 0.0;
    float3 found_pcd = make_float3(0.0, 0.0, 0.0);

    int64_t stack[STACK_SIZE]; stack[0] = 0; int stack_size = 1;

    int node = 0, depth = 0;
    do {

        const int3 childs = nodes[node];
        const int left_index = childs.x, right_index = childs.y;

        bool traverseL = false;
        float left_dist = 1e6;
        if (left_index >= 0 && depth < MAX_DEPTH){
            const float3 bbox_min_left = bbox[left_index * 2 + 0];
            const float3 bbox_max_left = bbox[left_index * 2 + 1];
            left_dist = bbox_dist(p, bbox_min_left, bbox_max_left);
            traverseL = left_dist < mindist;
        }

        bool traverseR = false;
        float right_dist = 1e6;
        if(right_index >= 0 && depth < MAX_DEPTH){
            const float3 bbox_min_right = bbox[right_index * 2 + 0];
            const float3 bbox_max_right = bbox[right_index * 2 + 1];
            right_dist = bbox_dist(p, bbox_min_right, bbox_max_right);
            traverseR = right_dist < mindist;
        }

        if ((left_index < 0 && right_index < 0) || (depth == MAX_DEPTH)) {
            // LEAF
            const int2 bounds = ranges[node];
            const int left_bound = bounds.x, right_bound = bounds.y;
            for(int tri_idx = left_bound; tri_idx <= right_bound; tri_idx++){
                const float3* curr_triangle = tri + tri_idx * 3;
                const float3 v0 = curr_triangle[0], v1 = curr_triangle[1], v2 = curr_triangle[2];

                // FIND MIN DISTANCE
                float tri_u, tri_v;
                const float dp = p_tri_dist(p, v0, v1, v2, tri_u, tri_v);
                if(dp < mindist) {
                    found_idx = tri_idx;
                    mindist = dp;
                    found_u = tri_u; found_v = tri_v;
                    found_pcd = v0 * (1 - tri_u - tri_v) + v1 * tri_u + v2 * tri_v;
                }
            }
        }


        if (!traverseL && !traverseR){
            int64_t item = stack[--stack_size];
            node = item >> 8; depth = item & 0xFFLL;
        }
        else if(traverseL && traverseR && stack_size < STACK_SIZE){
            node = (left_dist < right_dist) ? left_index : right_index;
            stack[stack_size++] = (static_cast<int64_t>((left_dist < right_dist) ? right_index : left_index) << 8) | (static_cast<int64_t>(++depth));
        }
        else {
            node = traverseL ? left_index : right_index;
            depth++;
        }

    } while(stack_size > 0);

    if (min_idx) *min_idx = found_idx;
    if (min_uv) *min_uv = make_float2(found_u, found_v);
    if (min_pcd) *min_pcd = found_pcd;

    return mindist;
}

__global__ void bvh_dist(
    const int n,
    const float3* __restrict__ q,
    const int3*  __restrict__ nodes,
    const int2* __restrict__ ranges,
    const float3* __restrict__ bbox,
    const float3* __restrict__ tri,
    float* __restrict__ dist,
    int* __restrict__ tri_idx,
    float2* __restrict__ tri_uv,
    float3* __restrict__ tri_pcd
){

    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx >= n)
        return;

    const float3 point = q[gidx];
    const float min_dist = pdist(point, nodes, ranges, bbox, tri, &tri_idx[gidx], &tri_uv[gidx], &tri_pcd[gidx]);

    dist[gidx] = min_dist;

}


__global__ void bvh_sign(
    const int n,
    const float3* __restrict__ q,
    const int3* __restrict__ nodes,
    const int2* __restrict__ ranges,
    const float3* __restrict__ bbox,
    const float3* __restrict__ tri,
    const int rand_rays,
    const int ext_rays,
    const float3* __restrict__ sphere_points,
    const bool check_parity,
    const float* __restrict__ dist,
    const float3* __restrict__ tri_pcd,
    const int* __restrict__ tri_idx,
    float* __restrict__ sgn
){

    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx >= n)
        return;

    const float3 point = q[gidx];
    
    // TODO validate normal based sign
//     if (tri_idx != nullptr && dist[gidx] < 0.0009765625f) {
//         int closest_tri_idx = tri_idx[gidx];
//
//         const float3* curr_triangle = tri + 3 * closest_tri_idx;
//         const float3 v0 = curr_triangle[0], v1 = curr_triangle[1], v2 = curr_triangle[2];
//
//         float3 nrm = normalize(cross(v1 - v0, v2 - v0));
//         float3 diff = tri_pcd[gidx] - point;
//
//         sgn[gidx] = (dot(diff, nrm) > 0.0f) ? -1.0f : 1.0f;
//         return;
//     }
    
    // Ray Casting Logic for further points
    const float cdist = fmaxf(fminf(dist[gidx] - FLT_EPSILON, 0.0009765625f), 0.0f);
    const float3 cpcd = tri_pcd[gidx];
    float3 surface_dir = normalize(cpcd - point);
    if (length(cpcd - point) < 1e-6f) surface_dir = make_float3(0.0f, 0.0f, 1.0f); // Default if on top

    // orthogonal to closest surface dir
    float3 base_dir = make_float3(0.0f, 0.0f, 1.0f);
    if(fabs(surface_dir.z) > 0.99f) base_dir = make_float3(0.0f, 1.0f, 0.0f);
    float3 dir1 = normalize(cross(surface_dir, base_dir));
    float3 dir2 = normalize(cross(dir1, surface_dir));

    int64_t stack[STACK_SIZE];

    const int n_rays = check_parity ? min(5, rand_rays) : rand_rays;
    int inside_count = 0;
    float point_sign = check_parity ? 1.0f : -1.0f;

    // #pragma unroll
    for(int dim = 0; dim < n_rays; dim++){

        const float3 target_point = normalize(sphere_points[dim]) * 2.0f + 0.5f;
        const float3 curr_dir = normalize(target_point - point);

        int intersections[5]; float3 points[5]; float3 dirs[5];
        points[0] = point;
        points[1] = point + dir1 * cdist; points[2] = point + dir2 * cdist;
        points[3] = point - dir1 * cdist; points[4] = point - dir2 * cdist;
        for(int k = 0; k < 5; k++) {
            intersections[k] = 0;
            dirs[k] = normalize(target_point - points[k]);
        }

        stack[0] = 0; int stack_size = 1;

        int node = 0, depth = 0;
        do {

            const int3 childs = nodes[node];
            const int left_index = childs.x;
            const int right_index = childs.y;

            bool traverseL = false;
            if (left_index >= 0 && depth < MAX_DEPTH){
                const float3 bbox_min_left = bbox[2 * left_index + 0];
                const float3 bbox_max_left = bbox[2 * left_index + 1];
                for(int k = 0; k < 5; k++) traverseL |= ray_bbox_intersect(points[k], dirs[k], bbox_min_left, bbox_max_left);
            }

            bool traverseR = false;
            if(right_index >= 0 && depth < MAX_DEPTH){
                const float3 bbox_min_right = bbox[2 * right_index + 0];
                const float3 bbox_max_right = bbox[2 * right_index + 1];
                for(int k = 0; k < 5; k++) traverseR |= ray_bbox_intersect(points[k], dirs[k], bbox_min_right, bbox_max_right);
            }

            if ((left_index < 0 && right_index < 0) || (depth == MAX_DEPTH)) {

                // LEAF
                const int2 bounds = ranges[node];
                const int left_bound = bounds.x, right_bound = bounds.y;

                for(int sorted_tri_idx = left_bound; sorted_tri_idx <= right_bound; sorted_tri_idx++){
                    const float3* curr_triangle = tri + 3 * sorted_tri_idx;
                    const float3 v0 = curr_triangle[0], v1 = curr_triangle[1], v2 = curr_triangle[2];
                    // CHECK INTERSECTION
                    float tri_depth, tri_u, tri_v; 
                    if(check_parity){
                        for(int k = 0; k < 5; k++) intersections[k] += (int) ray_tri_intersection(points[k], dirs[k], v0, v1, v2, tri_depth, tri_u, tri_v);
                    } else {
                        intersections[0] += (int) ray_tri_intersection(point, curr_dir, v0, v1, v2, tri_depth, tri_u, tri_v);
                    }

                }
            }


            if (!traverseL && !traverseR){
                int64_t item = stack[--stack_size];
                node = item >> 8; depth = item & 0xFFLL;
            }
            else {
                node = traverseL ? left_index : right_index;
                depth++;
                if(traverseL && traverseR && stack_size < STACK_SIZE) {
                    stack[stack_size++] = (static_cast<int64_t>(right_index) << 8) | static_cast<int64_t>(depth);
                }
            }

        } while((stack_size > 0) && (check_parity || intersections[0] == 0));

        if(check_parity){
            int inside_local = 0;
            for (int k = 0; k < 5; k++) inside_local += (intersections[k] % 2);
            inside_count += (int) (inside_local > 2);
            if (inside_count > n_rays / 2) {
                point_sign = -1.0f;
                break;
            }
        } else {
            inside_count += (int) (intersections[0] > 0);
            if(inside_count + ext_rays - 1 <= dim){
                point_sign = 1.0f;
                break;
            }
        }

    }

    sgn[gidx] = point_sign;

}

__device__ uint64_t expandBitsLong(uint64_t v){
    uint64_t x = v & 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

__device__ uint64_t morton3DLong(float x, float y, float z){
    x = clamp(rintf(x * 4096), 0.0f, (float) 4096 - 1);
    y = clamp(rintf(y * 4096), 0.0f, (float) 4096 - 1);
    z = clamp(rintf(z * 4096), 0.0f, (float) 4096 - 1);
    uint64_t xx = expandBitsLong((uint64_t) x);
    uint64_t yy = expandBitsLong((uint64_t) y);
    uint64_t zz = expandBitsLong((uint64_t) z);
    return (zz << 2) | (yy << 1) | xx;
}

__global__ void query_morton_kernel(const float* q, const int n, int64_t* codes){
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx < n){
        const float* curr_q = q + 3 * thread_idx;
        codes[thread_idx] = (int64_t) morton3DLong(curr_q[0], curr_q[1], curr_q[2]);
    }
}

sdf_out cuda_sdf(torch::Tensor q, torch::Tensor tri, bool use_sign, bool use_dist, bool check_parity, int rand_rays, int ext_rays) {
    const int n = q.size(0);
    const int threads = 128;

    bvh tree = build_tri_bvh(tri);
    auto sorted_tri = tri.index({tree.orig_idx});

    torch::Tensor q_codes = torch::empty(
        {n}, torch::TensorOptions().dtype(torch::kInt64).device(q.device())
    );
    query_morton_kernel<<<(n + threads - 1) / threads, threads>>>(
        q.data_ptr<float>(),
        n,
        q_codes.data_ptr<int64_t>()
    );
    torch::Tensor q_order = torch::argsort(q_codes);
    torch::Tensor q_sorted = q.index({q_order});


    // QUERIES TREE TRAVERSAL
    torch::Tensor dist = torch::empty(
        {n}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device())
    );
    torch::Tensor tri_idx = torch::empty(
        {n}, torch::TensorOptions().dtype(torch::kInt32).device(q.device())
    );
    torch::Tensor tri_uv = torch::empty(
        {n, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device())
    );
    torch::Tensor tri_pcd = torch::empty(
        {n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device())
    );

    if (use_dist){
        bvh_dist<<<(n + threads - 1) / threads, threads>>>(
            (int) n,
            reinterpret_cast<float3*>(q_sorted.data_ptr<float>()),
            reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
            reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
            reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
            reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
            dist.data_ptr<float>(),
            tri_idx.data_ptr<int>(),
            reinterpret_cast<float2*>(tri_uv.data_ptr<float>()),
            reinterpret_cast<float3*>(tri_pcd.data_ptr<float>())
        );
    } else {
        tri_idx.fill_(-1);
        dist.fill_(1.0f); tri_uv.fill_(0.0f); tri_pcd.fill_(0.0f);
    }


    torch::Tensor sign = torch::empty(
        {n}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device())
    );

    if (use_sign){
        torch::Tensor sphere_points = torch::randn({rand_rays, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
        bvh_sign<<<(n + threads - 1) / threads, threads>>>(
            (int) n,
              reinterpret_cast<float3*>(q_sorted.data_ptr<float>()),
            reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
            reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
            reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
            reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
            rand_rays,
            ext_rays,
            reinterpret_cast<float3*>(sphere_points.data_ptr<float>()),
            check_parity,
            dist.data_ptr<float>(),
            reinterpret_cast<float3*>(tri_pcd.data_ptr<float>()),
            use_dist ? tri_idx.data_ptr<int>() : nullptr,
            sign.data_ptr<float>()
        );
    } else {
        sign.fill_(1.0f);
    }

    if (use_dist) tri_idx = tree.orig_idx.index({tri_idx});

    torch::Tensor sdf = dist * sign;

    torch::Tensor inverse_index = torch::empty_like(q_order);
    inverse_index.index_put_({q_order}, torch::arange(0, q_order.size(0), torch::TensorOptions().dtype(q_order.dtype()).device(q_order.device())));

    return {sdf.index({inverse_index}), tri_idx.index({inverse_index}), tri_uv.index({inverse_index})};
}


inline __device__ int get_bit(int val, int num){
    return (int) ((val & (1 << num)) >> num);
}

inline __device__ int grid_idx(const int x, const int y, const int z, const int grid_res){
    return x * grid_res * grid_res + y * grid_res + z;
}


__global__ void up_tsdf_kernel(
    const int grid_res,
    const float* sdf,
    const float threshold,
    float* tsdf,
    bool *mask
){
    int gidx = global_idx();
    int n = grid_res * grid_res * grid_res;
    int coarse_res = grid_res / 2 + 1;

    if(gidx >= n) return;

    int x = gidx / (grid_res * grid_res);
    int y = (gidx / grid_res) % grid_res;
    int z = gidx % grid_res;


    bool near_surface = false;
    float sign = 0.0f;
    int pos = 0, neg = 0, close = 0;

    int tp = (x % 2) + (y % 2) + (z % 2);
    if (tp == 0){ // coarse grid point
        near_surface = false;
        sign = sdf[grid_idx(x / 2, y / 2, z / 2, coarse_res)];
    }
    else if(tp == 3){ // cube center
        for(int mask = 0; mask < 8; mask++){
            int xx = x / 2 + get_bit(mask, 0);
            int yy = y / 2 + get_bit(mask, 1);
            int zz = z / 2 + get_bit(mask, 2);
            const float val = sdf[grid_idx(xx, yy, zz, coarse_res)];
            sign += val; pos += (int) (val > 0.0f); neg += (int) (val < 0.0f); close += (int) (fabs(val) < threshold);
        }
        sign /= 8.0f;
        if (close > 0 || (pos > 0 && neg > 0)) near_surface = true;
    }
    else if (tp == 2) { // cube plane
        for(int pmask = 0; pmask < 3; pmask++){
            for(int mask = 0; mask < 4; mask++){
                int xx = x / 2, yy = y / 2, zz = z / 2;
                if(x % 2 == 0){
                    xx = clamp(xx + get_bit(pmask, 0) - get_bit(pmask, 1), 0, coarse_res - 1);
                    yy += get_bit(mask, 0);
                    zz += get_bit(mask, 1);
                }
                if(y % 2 == 0){
                    xx += get_bit(mask, 0);
                    yy = clamp(yy + get_bit(pmask, 0) - get_bit(pmask, 1), 0, coarse_res - 1);
                    zz += get_bit(mask, 1);
                }
                if(z % 2 == 0){
                    xx += get_bit(mask, 0);
                    yy += get_bit(mask, 1);
                    zz = clamp(zz + get_bit(pmask, 0) - get_bit(pmask, 1), 0, coarse_res - 1);
                }
                const float val = sdf[grid_idx(xx, yy, zz, coarse_res)];
                sign += val; pos += (int) (val > 0.0f); neg += (int) (val < 0.0f); close += (int) (fabs(val) < threshold);
            }
        }
        sign /= (3.0f * 4.0f);
        if (close > 0 || (pos > 0 && neg > 0)) near_surface = true;
    }
    else if (tp == 1){ // cube edge
        for(int pmask0 = 0; pmask0 < 3; pmask0++){
            for(int pmask1 = 0; pmask1 < 3; pmask1++){
                if(get_bit(pmask0, 0) + get_bit(pmask0, 1) + get_bit(pmask1, 0) + get_bit(pmask1, 1) > 1) continue;
                for(int mask = 0; mask < 2; mask++){
                    int xx = x / 2, yy = y / 2, zz = z / 2;
                    if(x % 2 == 1){
                        xx += get_bit(mask, 0);
                        yy = clamp(yy + get_bit(pmask0, 0) - get_bit(pmask0, 1), 0, coarse_res - 1);
                        zz = clamp(zz + get_bit(pmask1, 0) - get_bit(pmask1, 1), 0, coarse_res - 1);
                    }
                    if(y % 2 == 1){
                        xx = clamp(xx + get_bit(pmask0, 0) - get_bit(pmask0, 1), 0, coarse_res - 1);
                        yy += get_bit(mask, 0);
                        zz = clamp(zz + get_bit(pmask1, 0) - get_bit(pmask1, 1), 0, coarse_res - 1);
                    }
                    if(z % 2 == 1){
                        xx = clamp(xx + get_bit(pmask0, 0) - get_bit(pmask0, 1), 0, coarse_res - 1);
                        yy = clamp(yy + get_bit(pmask1, 0) - get_bit(pmask1, 1), 0, coarse_res - 1);
                        zz += get_bit(mask, 0);
                    }
                    const float val = sdf[grid_idx(xx, yy, zz, coarse_res)];
                    sign += val; pos += (int) (val > 0.0f); neg += (int) (val < 0.0f); close += (int) (fabs(val) < threshold);
                }
            }
        }
        sign /= (2.0f * 5.0f);
        if (close > 0 || (pos > 0 && neg > 0)) near_surface = true;
    }

    mask[gidx] = near_surface;
    if(!near_surface) tsdf[gidx] = sign;

}

std::vector<torch::Tensor> cuda_up_tsdf(torch::Tensor sdf, float threshold) {
    const int threads = 512;

    int grid_res = (int) std::round(std::pow((float) sdf.size(0), 1.0f / 3.0f));
    grid_res = 2 * grid_res - 1;

    int ncurr = grid_res * grid_res * grid_res;
    auto tsdf = torch::empty({ncurr}, torch::TensorOptions().dtype(torch::kFloat32).device(sdf.device()));
    auto mask = torch::empty({ncurr}, torch::TensorOptions().dtype(torch::kBool).device(sdf.device()));

    up_tsdf_kernel<<<(ncurr + threads - 1) / threads, threads>>>(
        grid_res,
        sdf.data_ptr<float>(),
        threshold,
        tsdf.data_ptr<float>(),
        mask.data_ptr<bool>()
    );

    return {tsdf, mask};
}


__device__ int3 gidx2coord(int gidx, int GRID_RES){
    // decode gidx to coordinates
    int rem = gidx;
    int z = rem % (GRID_RES + 1);
    rem /= (GRID_RES + 1);
    int y = rem % (GRID_RES + 1);
    rem /= (GRID_RES + 1);
    int x = rem;
    return make_int3(x, y, z);
}

// #define GRID_DIRS 6 // 2 * 3
#define GRID_DIRS 26 // 3 * 3 * 3 - 1

__device__ int3 dir2adj(const int3 &coord, int dir_idx){
    #if GRID_DIRS == 6
        int side = dir_idx / 2, dir = dir_idx % 2;
        int xx = coord.x + static_cast<int>(side == 0) * (1 - dir * 2);
        int yy = coord.y + static_cast<int>(side == 1) * (1 - dir * 2);
        int zz = coord.z + static_cast<int>(side == 2) * (1 - dir * 2);
    #elif GRID_DIRS == 26
        dir_idx += 1; // skip no move
        int x_dir = 1 - dir_idx % 3;
        int y_dir = 1 - (dir_idx / 3) % 3;
        int z_dir = 1 - (dir_idx / 9) % 3;
        int xx = coord.x + x_dir, yy = coord.y + y_dir, zz = coord.z + z_dir;
    #endif
    return make_int3(xx, yy, zz);
}

__device__ __forceinline__ int coord2gidx(const int x, const int y, const int z, const int GRID_RES)
{
    return (x * (GRID_RES + 1) + y) * (GRID_RES + 1) + z;
}

template<int WPT>
__global__ void init_grid_graph_kernel(
    const int GRID_RES,
    const int N_MID,
    const int N_LOW,
    const int WORK_PER_THREAD,
    const int3* __restrict__ nodes,
    const int2* __restrict__ ranges,
    const float3* __restrict__ bbox,
    const float3* __restrict__ tri,
    float* __restrict__ dist,
    int8_t* __restrict__ cap,
    bool* __restrict__ near
) {
    const int group_idx = blockIdx.x;
    const int THREADS = blockDim.x;
    const int local_idx = threadIdx.x;

    const int low_idx = group_idx % N_LOW;
    const int mid_idx = (group_idx / N_LOW) % N_MID;
    const int hgh_idx = group_idx / (N_LOW * N_MID);

    // round_up of mid items to process
    const int3 base_coord = make_int3(
        hgh_idx,
        mid_idx * WORK_PER_THREAD,
        low_idx * THREADS + local_idx
    );
    if (base_coord.z > GRID_RES) return; // free extra threads

    const int ACTIVE_WORK_PER_THREAD = min(WORK_PER_THREAD, GRID_RES + 1 - base_coord.y);

    float mindist[WPT]; float3 pnts[WPT];
    #pragma unroll
    for (size_t k = 0; k < WPT; k++)
    {
        mindist[k] = FLT_MAX;
        pnts[k] = make_float3(
            static_cast<float>(base_coord.x) / GRID_RES,
            static_cast<float>(base_coord.y + k) / GRID_RES,
            static_cast<float>(base_coord.z) / GRID_RES
        );
    }


    constexpr int TREE_DEPTH = 48;
    constexpr int TREE_STACK = (TREE_DEPTH * 2);

    int64_t stack[TREE_STACK]; stack[0] = 0; int stack_size = 1;
    int node = 0, depth = 0;
    do {


        const int3 childs = nodes[node];

        bool traverseL = false;
        float left_dist = FLT_MAX;
        if (childs.x >= 0 && depth < TREE_DEPTH){
            const float3 bbox_min_left = bbox[childs.x * 2 + 0];
            const float3 bbox_max_left = bbox[childs.x * 2 + 1];
            // #pragma unroll
            for (size_t k = 0; k < WPT; k++)
            {
                if (k < ACTIVE_WORK_PER_THREAD)
                {
                    const float _bbox_dist = bbox_dist(pnts[k], bbox_min_left, bbox_max_left);
                    left_dist = fminf(_bbox_dist, left_dist);
                    traverseL |= _bbox_dist < mindist[k];
                }
            }
        }

        bool traverseR = false;
        float right_dist = FLT_MAX;
        if(childs.y >= 0 && depth < TREE_DEPTH){
            const float3 bbox_min_right = bbox[childs.y * 2 + 0];
            const float3 bbox_max_right = bbox[childs.y * 2 + 1];
            // #pragma unroll
            for (size_t k = 0; k < WPT; k++)
            {
                if (k < ACTIVE_WORK_PER_THREAD)
                {
                    const float _bbox_dist = bbox_dist(pnts[k], bbox_min_right, bbox_max_right);
                    right_dist = fminf(_bbox_dist, right_dist);
                    traverseR |= _bbox_dist < mindist[k];
                }
            }
        }

        if ((childs.x < 0 && childs.y < 0) || (depth == TREE_DEPTH)) {
            // LEAF
            const int2 bounds = ranges[node];
            for(int tri_idx = bounds.x; tri_idx <= bounds.y; tri_idx++){
                const float3* curr_triangle = tri + tri_idx * 3;
                const float3 v0 = curr_triangle[0], v1 = curr_triangle[1], v2 = curr_triangle[2];

                // FIND MIN DISTANCE
                float tri_u, tri_v;
                // #pragma unroll
                for (size_t k = 0; k < WPT; k++)
                {
                    if (k < ACTIVE_WORK_PER_THREAD)
                    {
                        mindist[k] = fminf(p_tri_dist(pnts[k], v0, v1, v2, tri_u, tri_v), mindist[k]);
                    }
                }
            }
        }

        if (!traverseL && !traverseR){
            int64_t item = stack[--stack_size];
            node = item >> 8; depth = item & 0xFFLL;
        }
        else if(traverseL && traverseR && stack_size < TREE_STACK){
            node = (left_dist < right_dist) ? childs.x : childs.y;
            stack[stack_size++] = (static_cast<int64_t>((left_dist < right_dist) ? childs.y : childs.x) << 8) | (static_cast<int64_t>(++depth));
        }
        else {
            node = traverseL ? childs.x : childs.y;
            depth++;
        }

    } while(stack_size > 0);

    #pragma unroll
    for (size_t k = 0; k < WPT; k++)
    {
        if (k < ACTIVE_WORK_PER_THREAD)
            dist[coord2gidx(base_coord.x, base_coord.y + k, base_coord.z, GRID_RES)] = mindist[k];
    }

    const int N_CORNERS = (GRID_RES + 1) * (GRID_RES + 1) * (GRID_RES + 1);
    const float shift = 1.0f / static_cast<float>(GRID_RES);

    bool is_near[WPT];
    #pragma unroll
    for (size_t k = 0; k < WPT; k++) is_near[k] = false;

    // #pragma unroll
    for(int dir_idx = 0; dir_idx < GRID_DIRS; dir_idx++){

        // #pragma unroll
        for (size_t k = 0; k < WPT; k++)
        {
            if (k < ACTIVE_WORK_PER_THREAD && mindist[k] > shift)
            {
                const int3 coord = make_int3(base_coord.x, base_coord.y + k, base_coord.z);
                const int3 adj = dir2adj(coord, dir_idx);
                if ((adj.x >= 0 && adj.y >= 0 && adj.z >= 0) && (adj.x <= GRID_RES && adj.y <= GRID_RES && adj.z <= GRID_RES))
                {
                    const float3 pp = make_float3(static_cast<float>(adj.x) / GRID_RES, static_cast<float>(adj.y) / GRID_RES, static_cast<float>(adj.z) / GRID_RES);
                    if (mindist[k] > length(pnts[k] - pp))
                        cap[coord2gidx(coord.x, coord.y, coord.z, GRID_RES)] = static_cast<int8_t>(1);
                    else is_near[k] = true;
                }
            }
        }

        cap += N_CORNERS;
    }

    #pragma unroll
    for (size_t k = 0; k < WPT; k++)
        if (k < ACTIVE_WORK_PER_THREAD && is_near[k])
            near[coord2gidx(base_coord.x, base_coord.y + k, base_coord.z, GRID_RES)] = true;

}

__global__ void near_grid_graph_kernel(
    const int near_size,
    const int* __restrict__ near_idx,
    const int GRID_RES,
    const int3* __restrict__ nodes,
    const int2* __restrict__ ranges,
    const float3* __restrict__ bbox,
    const float3* __restrict__ tri,
    const float* __restrict__ dist,
    int8_t* __restrict__ cap
) {

    const int tidx = global_idx();
    if (tidx >= near_size) return;

    const int gidx = near_idx[tidx];
    const int N_CORNERS = (GRID_RES + 1) * (GRID_RES + 1) * (GRID_RES + 1);
    const int3 coord = gidx2coord(gidx, GRID_RES);

    const float3 p = make_float3(coord.x / (float) GRID_RES, coord.y / (float) GRID_RES, coord.z / (float) GRID_RES);
    float d = dist[gidx];

    const float shift = 1.0f / (float) GRID_RES;
    for (int dir_idx = 0; dir_idx < GRID_DIRS; dir_idx++){

        if (dir_idx > 0) cap += N_CORNERS;

        const int3 adj = dir2adj(coord, dir_idx);
        if(
            adj.x < 0 || adj.y < 0 || adj.z < 0 || adj.x > GRID_RES || adj.y > GRID_RES || adj.z > GRID_RES
        ) continue;

        const float3 pp = make_float3(adj.x / (float) GRID_RES, adj.y / (float) GRID_RES, adj.z / (float) GRID_RES);
        if (d > length(p - pp)) continue;


        float3 pcurr = p; float3 vec = normalize(pp - p);
        float  dcurr = d;
        bool blocked = false; int nsteps = 0;
        while(!blocked){
            pcurr = pcurr + vec * clamp(length(pp - pcurr) - shift, FLT_EPSILON, dcurr);
            dcurr = pdist(pcurr, nodes, ranges, bbox, tri, nullptr, nullptr, nullptr);
            blocked |= dcurr <= shift;
            blocked |= (++nsteps > 32);
            if(dcurr > length(pp - pcurr)) break;
        }
        if(blocked) continue;

        cap[gidx] = static_cast<int8_t>(1);

    }
}

__global__ void init_dsu_kernel(
    const int GRID_RES,
    int32_t* __restrict__ dsu
) {

    const int gidx = global_idx();
    const int N_CORNERS = (GRID_RES + 1) * (GRID_RES + 1) * (GRID_RES + 1);
    if (gidx > N_CORNERS) return;

    const int3 coord = gidx2coord(gidx, GRID_RES);
    if (
        coord.x == 0 || coord.x == GRID_RES ||
        coord.y == 0 || coord.y == GRID_RES ||
        coord.z == 0 || coord.z == GRID_RES
    ) {
        dsu[gidx] = N_CORNERS;
    } else {
        dsu[gidx] = gidx;
    }

}

__global__ void merge_voxels_kernel(
    const int GRID_RES,
    int8_t* __restrict__ cap,
    int32_t* __restrict__ dsu
) {

    const int gidx = global_idx();
    const int N_CORNERS = (GRID_RES + 1) * (GRID_RES + 1) * (GRID_RES + 1);
    if (gidx >= N_CORNERS) return;

    const int3 coord = gidx2coord(gidx, GRID_RES);
    for (int dir_idx = 0; dir_idx < GRID_DIRS; dir_idx++){

        if(cap[gidx] > 0){
            const int3 adj = dir2adj(coord, dir_idx);
            int target_idx = (adj.x * (GRID_RES + 1) + adj.y) * (GRID_RES + 1) + adj.z;
            dsu_union(gidx, target_idx, dsu);
        }

        cap += N_CORNERS;
    }

}

__global__ void finalize_dsu_kernel(
    const int N_CORNERS,
    int32_t* __restrict__ dsu,
    float* __restrict__ sgn
) {

    const int gidx = global_idx();
    if (gidx >= N_CORNERS) return;

    dsu_find(gidx, dsu); // make tree depth 1

    if (dsu[gidx] == N_CORNERS){
        sgn[gidx] = 1.0;
    } else {
        sgn[gidx] = -1.0;
    }

}

torch::Tensor cuda_flow_sdf(torch::Tensor tri, const int GRID_RES){

    auto device = tri.device();

    bvh tree = build_tri_bvh(tri);
    auto sorted_tri = tri.index({tree.orig_idx});

    const int N_CORNERS = (GRID_RES + 1) * (GRID_RES + 1) * (GRID_RES + 1);
    auto dist = torch::empty({GRID_RES + 1, GRID_RES + 1, GRID_RES + 1}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    auto cap = torch::zeros(
        {GRID_DIRS, GRID_RES + 1, GRID_RES + 1, GRID_RES + 1},
        torch::TensorOptions().device(device).dtype(torch::kInt8)
    );
    auto near = torch::zeros({N_CORNERS}, torch::TensorOptions().device(device).dtype(torch::kBool));

    int threads = 1;
    while ((threads << 1) <= (GRID_RES + 1) && threads < 128) threads <<= 1;
    const int WORK_PER_THREAD = min(GRID_RES + 1, 4);
    const int N_MID = (GRID_RES + 1 + WORK_PER_THREAD - 1) / WORK_PER_THREAD;
    const int N_LOW = (GRID_RES + 1 + threads - 1) / threads;
    init_grid_graph_kernel<4><<<(GRID_RES + 1) * N_MID * N_LOW, threads>>>(
        GRID_RES,
        N_MID,
        N_LOW,
        WORK_PER_THREAD,
        reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
        reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
        reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
        reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
        dist.data_ptr<float>(),
        cap.data_ptr<int8_t>(),
        near.data_ptr<bool>()
    );

    threads = 256;
    dim3 gridSize((N_CORNERS + threads - 1) / threads), blockSize(threads);

    auto near_idx = torch::arange(0, N_CORNERS, torch::TensorOptions().device(device).dtype(torch::kInt32)).index({near});
    int near_size = near_idx.size(0);
    if (near_size > 0){
        near_grid_graph_kernel<<<(near_size + threads - 1) / threads, threads>>>(
            near_size,
            near_idx.data_ptr<int32_t>(),
            GRID_RES,
            reinterpret_cast<int3*>(tree.nodes.data_ptr<int>()),
            reinterpret_cast<int2*>(tree.ranges.data_ptr<int>()),
            reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
            reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
            dist.data_ptr<float>(),
            cap.data_ptr<int8_t>()
        );
    }


    auto dsu = torch::empty({N_CORNERS + 1}, torch::TensorOptions().device(device).dtype(torch::kInt32));
    init_dsu_kernel<<<(N_CORNERS + 1 + threads - 1) / threads, blockSize>>>(
        GRID_RES,
        dsu.data_ptr<int32_t>()
    );
    merge_voxels_kernel<<<gridSize, blockSize>>>(
        GRID_RES,
        cap.data_ptr<int8_t>(),
        dsu.data_ptr<int32_t>()
    );

    auto sign = torch::empty({GRID_RES + 1, GRID_RES + 1, GRID_RES + 1}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    finalize_dsu_kernel<<<gridSize, blockSize>>>(
        N_CORNERS,
        dsu.data_ptr<int32_t>(),
        sign.data_ptr<float>()
    );

    dist.mul_(sign);
    return dist;
}
