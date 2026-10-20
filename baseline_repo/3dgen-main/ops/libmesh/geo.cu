

#include <mesh.h>
#include <bvh.h>
#include <cutil_math.h>

__device__ __forceinline__ double tri_sphere_area(
    const double3 &tri0, const double3 &tri1, const double3 &tri2,
    const double3 &center, const float radius
) {

    double3 u = tri1 - tri0, v = tri2 - tri0;
    double3 n = cross(u, v);
    if (length(n) < 1e-12) return 0.0;
    n = normalize(n);

    double d = dot(n, (center - tri0));
    if (fabs(d) >= static_cast<double>(radius)) return 0.0;

    const double3 pcenter = center - d * n;
    const double r = sqrt(static_cast<double>(radius) * static_cast<double>(radius) - d * d);

    if (length(u) < 1e-8) {
        if (length(v) < 1e-8) return 0.0;
        u = v;
    }
    u = normalize(u);
    v = cross(n, u);

    double2 tri[3];
    tri[0] = make_double2(dot(tri0 - pcenter, u), dot(tri0 - pcenter, v));
    tri[1] = make_double2(dot(tri1 - pcenter, u), dot(tri1 - pcenter, v));
    tri[2] = make_double2(dot(tri2 - pcenter, u), dot(tri2 - pcenter, v));

    double area = 0.0;
    for(size_t k = 0; k < 3; k++){

        double2 p1 = tri[k];
        double2 p2 = tri[(k + 1) % 3];
        double2 dp = p2 - p1;

        double a = dot(dp, dp), b = 2.0 * dot(p1, dp), c = dot(p1, p1) - r * r;
        double det = b * b - 4.0 * a * c;

        double2 pts[4]; int pn = 0;
        pts[pn++] = p1;
        if (det > 1e-8 && length(dp) > 1e-6) {
            det = sqrt(det);
            double t1 = (-b - det) / (2.0 * a);
            if (t1 > 1e-8 && t1 < 1.0 - 1e-8) pts[pn++] = p1 + t1 * dp;
            double t2 = (-b + det) / (2.0 * a);
            if (t2 > 1e-8 && t2 < 1.0 - 1e-8) pts[pn++] = p1 + t2 * dp;
        }
        pts[pn++] = p2;

        for(size_t j = 0; j < pn - 1; j++){
            double2 q1 = pts[j], q2 = pts[j + 1];
            double2 mid = (q1 + q2) * 0.5;
            if (dot(mid, mid) <= r * r) {
                area += 0.5 * (q1.x * q2.y - q1.y * q2.x);
            }else{
                area += 0.5 * r * r * atan2(q1.x * q2.y - q1.y * q2.x, dot(q1, q2));
            }
        }
    }

    return fabs(area);

}

#define PI 3.1415926f
#define BINS 64
#define NFEATURES (BINS * 4 + 2)

__global__ void faggregate_kernel(
    const int n_faces,
    const int* f,
    const float farea,
    const float* v,
    float* vnrm,
    float* vangle,
    const int n_points,
    const float *points,
    const float *normals,
    const float dist,
    float* buffer
){
    const int gidx = global_idx();
    if (gidx >= n_faces)
        return;

    int vi[3]; double3 vert[3];
    for(int k = 0; k < 3; k++){
        vi[k] = f[gidx * 3 + k];
        vert[k] = make_double3(static_cast<double>(v[3 * vi[k] + 0]), static_cast<double>(v[3 * vi[k] + 1]), static_cast<double>(v[3 * vi[k] + 2]));
    }

    double3 fnrm = cross(vert[1] - vert[0], vert[2] - vert[0]);
    double area = length(fnrm) / 2.0f;
    if (area < 1e-12){
        fnrm = make_double3(0.0, 0.0, 1.0);
    }else{
        fnrm = normalize(fnrm);
    }


    for(int k = 0; k < 3; k++){

        atomicAdd(vnrm + 3 * vi[k] + 0, static_cast<float>(fnrm.x));
        atomicAdd(vnrm + 3 * vi[k] + 1, static_cast<float>(fnrm.y));
        atomicAdd(vnrm + 3 * vi[k] + 2, static_cast<float>(fnrm.z));

        double angle_dot = dot(normalize(vert[(k + 1) % 3] - vert[k]), normalize(vert[(k + 2) % 3] - vert[k]));
        float angle = static_cast<float>(acos(fmin(fmax(angle_dot, -1.0 + 1e-6), 1.0 - 1e-6)));
        atomicAdd(vangle + vi[k], clamp(angle, 0.0f, PI));
    }

    for(int vidx = 0; vidx < n_points; vidx++){

        const double3 v1 = make_double3(static_cast<double>(points[3 * vidx + 0]), static_cast<double>(points[3 * vidx + 1]), static_cast<double>(points[3 * vidx + 2]));
        const double3 nrm1 = make_double3(static_cast<double>(normals[3 * vidx + 0]), static_cast<double>(normals[3 * vidx + 1]), static_cast<double>(normals[3 * vidx + 2]));

        const double area = tri_sphere_area(vert[0], vert[1], vert[2], v1, dist);
        if(area < 1e-12) continue;
        atomicAdd(buffer + vidx * NFEATURES + BINS * 4 + 1, static_cast<float>(area));

        const float pdot = static_cast<float>(dot(fnrm, normalize(nrm1)));
        int dbin = clamp(floorf((pdot + 1.0f) / 2.0f * static_cast<float>(BINS)), 0.0f, static_cast<float>(BINS - 1));
        atomicAdd(buffer + vidx * NFEATURES + BINS * 0 + max(0, min(dbin, BINS - 1)), 1.0f);

    }
}


__global__ void vaggregate_kernel(
    const int n_vertices,
    const float* v,
    const float* vnrm,
    const float* vcurve,
    const int n_points,
    const float *points,
    const float *normals,
    const float dist,
    float* buffer
){

    const int gidx = global_idx();
    if (gidx >= n_vertices)
        return;

    const float3 v0 = make_float3(v[3 * gidx + 0], v[3 * gidx + 1], v[3 * gidx + 2]);
    float3 nrm0 = make_float3(vnrm[3 * gidx + 0], vnrm[3 * gidx + 1], vnrm[3 * gidx + 2]);
    if(length(nrm0) < 1e-8f) {
        nrm0 = make_float3(0.0, 0.0, 1.0);
    }
    nrm0 = normalize(nrm0);

    float curve = (2.0f * PI - vcurve[gidx]);
    int cbin = clamp(floorf(((curve / (2.0f * PI)) / 2.0f + 1.0f) * static_cast<float>(2 * BINS)), 0.0f, static_cast<float>(2 * BINS - 1));

    for (int vidx = 0; vidx < n_points; vidx++){

        const float3 v1 = make_float3(points[3 * vidx + 0], points[3 * vidx + 1], points[3 * vidx + 2]);
        const float3 nrm1 = make_float3(normals[3 * vidx + 0], normals[3 * vidx + 1], normals[3 * vidx + 2]);

        if(length(v0 - v1) > dist) continue;

        float pdot = dot(nrm0, normalize(nrm1));
        int dbin = clamp(floorf((pdot + 1.0f) / 2.0f * static_cast<float>(BINS)), 0.0f, static_cast<float>(BINS - 1));
        atomicAdd(buffer + vidx * NFEATURES + BINS * 1 + max(0, min(dbin, BINS - 1)), 1.0f);

        atomicAdd(buffer + vidx * NFEATURES + BINS * 2 + max(0, min(cbin, 2 * BINS - 1)), 1.0f);
        atomicAdd(buffer + vidx * NFEATURES + BINS * 4 + 0, 1.0f);
    }

}

torch::Tensor cuda_local_features(torch::Tensor base_pcd, torch::Tensor base_nrm, float dist, torch::Tensor v, torch::Tensor f, float farea) {

    const int threads = 512;

    const int n_points = base_pcd.size(0);
    const int n_vertices = v.size(0), n_faces = f.size(0);

    auto features = torch::zeros(
        {n_points, NFEATURES},
        torch::TensorOptions().dtype(torch::kFloat32).device(base_pcd.device())
    );

    auto vnrm = torch::zeros(
        {n_vertices, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(base_pcd.device())
    );
    auto vangle = torch::zeros(
        {n_vertices},
        torch::TensorOptions().dtype(torch::kFloat32).device(base_pcd.device())
    );
    faggregate_kernel<<<(n_faces + threads - 1) / threads, threads>>>(
        n_faces,
        f.data_ptr<int>(),
        farea,
        v.data_ptr<float>(),
        vnrm.data_ptr<float>(),
        vangle.data_ptr<float>(),
        n_points,
        base_pcd.data_ptr<float>(),
        base_nrm.data_ptr<float>(),
        dist,
        features.data_ptr<float>()
    );

    vaggregate_kernel<<<(n_vertices + threads - 1) / threads, threads>>>(
        n_vertices,
        v.data_ptr<float>(),
        vnrm.data_ptr<float>(),
        vangle.data_ptr<float>(),
        n_points,
        base_pcd.data_ptr<float>(),
        base_nrm.data_ptr<float>(),
        dist,
        features.data_ptr<float>()
    );


    return features;
}

#define MAX_PROP_STEPS 16384
#define FACE_PER_THREAD 128
#define QSIZE 128
#define THREAD_STEPS 1024

__global__ void face_propagate_kernel(
    const int n_faces,
    const int *face2face,
    int *labels
){

    const int tidx = global_idx();

    int queue[QSIZE], qsize = 0, steps = 0;
    for(size_t local_idx = 0; local_idx < FACE_PER_THREAD; local_idx++){

        const int fidx = tidx * FACE_PER_THREAD + local_idx;
        if (fidx >= n_faces) return;

        int label = labels[fidx]; bool is_minimal = true;
        qsize = 0; queue[qsize++] = fidx;

        while(is_minimal && qsize > 0 && steps++ < THREAD_STEPS){

            int idx = queue[--qsize];
            for(int k = 0; k < 3; k++) {
                int next_idx = face2face[idx * 3 + k];
                int old_label = atomicMin(labels + next_idx, label);
                if (old_label < label) {
                    label = old_label; is_minimal = false;
                    queue[qsize++] = idx; // return back to reset
                    break;
                } else if (old_label > label && qsize < QSIZE) queue[qsize++] = next_idx;
            }

        }

        // we found label smaller than current do reverse path
        while (!is_minimal && qsize > 0) {
            label = min(atomicMin(labels + queue[--qsize], label), label);
        }

    }

}

void cuda_face_propagate(torch::Tensor labels, torch::Tensor face2face){

    const int threads = 128;
    int n_faces = labels.size(0);

    auto old_labels = labels.clone();
    for(size_t step = 0; step < MAX_PROP_STEPS; step++){
        face_propagate_kernel<<<(n_faces + threads * FACE_PER_THREAD - 1) / (threads * FACE_PER_THREAD), threads>>>(
            n_faces,
            face2face.data_ptr<int>(),
            labels.data_ptr<int>()
        );
        if (labels.equal(old_labels)) break;
        old_labels.copy_(labels);
    }

}



__global__ void build_oriented_edge_code_kernel(
    const int N_FACES,
    const int3* __restrict__ faces,
    int64_t* __restrict__ codes
) {

    const int gidx = global_idx();
    if(gidx >= N_FACES) return;

    const int3 f = faces[gidx];
    int fids[3] = {f.x, f.y, f.z};

    #pragma unroll
    for(size_t k = 0; k < 3; k++) {
        const int64_t i0 = static_cast<int64_t>(fids[k]);
        const int64_t i1 = static_cast<int64_t>(fids[(k + 1) % 3]);
        codes[gidx * 3 + k] = (i0 << 31) | i1;
    }

}

__global__ void build_oriented_face_graph(
    const int N_FACES,
    const int3* __restrict__ faces,
    const int N_VERTS,
    const float3* __restrict__ verts,
    const float thr,
    const int64_t* __restrict__ codes,
    const int32_t* __restrict__ code_map,
    int32_t* __restrict__ face_edges
) {

    const int gidx = global_idx();
    if(gidx >= N_FACES) return;

    const int3 f = faces[gidx];
    int fids[3] = {f.x, f.y, f.z};

    float3 v0 = verts[f.x], v1 = verts[f.y], v2 = verts[f.z];
    float3 nrm0 = normalize(cross(v1 - v0, v2 - v0));

    int ecnt = 0;
    #pragma unroll
    for(size_t k = 0; k < 3; k++) {

        const int64_t i0 = static_cast<int64_t>(fids[k]);
        const int64_t i1 = static_cast<int64_t>(fids[(k + 1) % 3]);

        const int64_t inverse_edge = (i1 << 31) | i0;

        int l = 0, r = 3 * N_FACES;
        while(r - l > 1){
            int m = (l + r) >> 1;
            if (codes[m] <= inverse_edge) l = m;
            else r = m;
        }
        // l is rightmost found
        for(int step = 0; step < 3; step++){
            if (l - step < 0) break;
            if (codes[l - step] != inverse_edge) break;
            int face_idx = code_map[l - step] / 3;

            const int3 ff = faces[face_idx];
            float3 nrm1 = normalize(cross(verts[ff.y] - verts[ff.x], verts[ff.z] - verts[ff.x]));
            if (acosf(fminf(fmaxf(dot(nrm0, nrm1), -1.0f + 1e-8f), 1.0f - 1e-8f)) > thr) continue;

            face_edges[gidx * 9 + ecnt++] = face_idx;
        }

    }

}


__global__ void merge_faces_kernel(
    const int N_FACES,
    const int32_t* __restrict__ face_edges,
    int32_t* __restrict__ label
){

    const int gidx = global_idx();
    if(gidx >= N_FACES) return;

    for(int eidx = 0; eidx < 9; eidx++){
        int face_idx = face_edges[gidx * 9 + eidx];
        if (face_idx < 0) break;
        dsu_union(gidx, face_idx, label);
    }

}

__global__ void dsu_compact_kernel(
    const int N,
    int32_t* __restrict__ dsu
) {

    const int gidx = global_idx();
    if (gidx >= N) return;

    dsu_find(gidx, dsu);

}

torch::Tensor cuda_oriented_components(torch::Tensor verts, torch::Tensor faces, const float thr) {

    const int threads = 256;
    auto device = faces.device();

    const int N_VERTS = verts.size(0), N_FACES = faces.size(0);
    auto edge_code = torch::empty({3 * N_FACES}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    build_oriented_edge_code_kernel<<<(N_FACES + threads - 1) / threads, threads>>>(
        N_FACES,
        reinterpret_cast<int3*>(faces.data_ptr<int32_t>()),
        edge_code.data_ptr<int64_t>()
    );

    auto _edge_code = torch::sort(edge_code);
    edge_code = std::get<0>(_edge_code);
    auto edge_code_map = std::get<1>(_edge_code).to(torch::kInt32);

    auto face_edges = torch::full({N_FACES * 9}, -1, torch::TensorOptions().dtype(torch::kInt32).device(device));
    build_oriented_face_graph<<<(N_FACES + threads - 1) / threads, threads>>>(
        N_FACES,
        reinterpret_cast<int3*>(faces.data_ptr<int32_t>()),
        N_VERTS,
        reinterpret_cast<float3*>(verts.data_ptr<float>()),
        thr,
        edge_code.data_ptr<int64_t>(),
        edge_code_map.data_ptr<int32_t>(),
        face_edges.data_ptr<int32_t>()
    );

    auto face_label = torch::arange(0, N_FACES, torch::TensorOptions().dtype(torch::kInt32).device(device));
    merge_faces_kernel<<<(N_FACES + threads - 1) / threads, threads>>>(
        N_FACES,
        face_edges.data_ptr<int32_t>(),
        face_label.data_ptr<int32_t>()
    );
    dsu_compact_kernel<<<(N_FACES + threads - 1) / threads, threads>>>(N_FACES, face_label.data_ptr<int32_t>());


    return face_label;
}


__device__ bool tri_bbox_intersect(
    const float3 tri0, const float3 tri1, const float3 tri2,
    const float3 bb_min, const float3 bb_max
){

    const float3 tri_min = fminf(fminf(tri0, tri1), tri2), tri_max = fmaxf(fmaxf(tri0, tri1), tri2);
    if ((tri_min.x > bb_max.x || tri_max.x < bb_min.x) ||
        (tri_min.y > bb_max.y || tri_max.y < bb_min.y) ||
        (tri_min.z > bb_max.z || tri_max.z < bb_min.z)) return false;

    const float3 center = (bb_min + bb_max) / 2.0f;
    const float3 bb_half = bb_max - center;
    const float3 v0 = tri0 - center, v1 = tri1 - center, v2 = tri2 - center;

    float3 edges[3];
    edges[0] = v1 - v0;
    edges[1] = v2 - v1;
    edges[2] = v0 - v2;

    const float3 nrm = cross(edges[0], edges[1]);
    const float dist = fabsf(dot(nrm, v0));
    const float bb_radius = dot(bb_half, fabs(nrm));
    if(dist > bb_radius) return false;

    float3 axes[13];
    axes[0] = make_float3(1.0f, 0.0f, 0.0f);
    axes[1] = make_float3(0.0f, 1.0f, 0.0f);
    axes[2] = make_float3(0.0f, 0.0f, 1.0f);
    axes[3] = nrm;

    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            axes[4 + i * 3 + j] = cross(edges[i], axes[j]);
        }
    }

    for(size_t i = 0; i < 13; i++){
        if (length(axes[i]) < 1e-8f) continue;
        const float p0 = dot(v0, axes[i]), p1 = dot(v1, axes[i]), p2 = dot(v2, axes[i]);
        const float r = dot(bb_half, fabs(axes[i]));
        if (fminf(fminf(p0, p1), p2) > r || fmaxf(fmaxf(p0, p1), p2) < -r) return false;
    }

    return true;
}


__device__ bool tri_tri_intersect(
    const float3 v0, const float3 v1, const float3 v2,
    const float3 u0, const float3 u1, const float3 u2,
    const float tol
){

    const float3 t1min = fminf(fminf(v0, v1), v2), t1max = fmaxf(fmaxf(v0, v1), v2);
    const float3 t2min = fminf(fminf(u0, u1), u2), t2max = fmaxf(fmaxf(u0, u1), u2);

    if ((t1min.x > t2max.x) || (t1max.x < t2min.x) ||
        (t1min.y > t2max.y) || (t1max.y < t2min.y) ||
        (t1min.z > t2max.z) || (t1max.z < t2min.z)) return false;


    float3 e1[3];
    e1[0] = v1 - v0;
    e1[1] = v2 - v1;
    e1[2] = v0 - v2;

    float3 e2[3];
    e2[0] = u1 - u0;
    e2[1] = u2 - u1;
    e2[2] = u0 - u2;

    const int n_axes = 2 + 3 * 2 + 3 * 3;
    float3 axes[n_axes];

    // base normals
    axes[0] = cross(e1[0], e1[1]);
    axes[1] = cross(e2[0], e2[1]);

    // edge normals
    for(size_t k = 0; k < 3; k++){
        axes[2 + k] = cross(axes[0], e1[k]);
    }
    for(size_t k = 0; k < 3; k++){
        axes[2 + 3 + k] = cross(axes[1], e2[k]);
    }

    // crosses
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++) {
            axes[2 + 3 * 2 + i * 3 + j] = cross(e1[i], e2[j]);
        }
    }


    for(size_t i = 0; i < n_axes; i++){
        if (length(axes[i]) < 1e-12f) continue;
        float3 axis = normalize(axes[i]);
        const float p0 = dot(v0, axis), p1 = dot(v1, axis), p2 = dot(v2, axis);
        const float f0 = dot(u0, axis), f1 = dot(u1, axis), f2 = dot(u2, axis);
        if (
            (fminf(fminf(p0, p1), p2) > fmaxf(fmaxf(f0, f1), f2) - tol) ||
            (fminf(fminf(f0, f1), f2) > fmaxf(fmaxf(p0, p1), p2) - tol)
        ) return false;
    }


    return true;
}

#define BVH_STACK_SIZE 128
#define BVH_MAX_DEPTH 32

__device__ int tri_intersects(
    const float3 v0,
    const float3 v1,
    const float3 v2,
    const int gidx,
    const int *nodes,
    const int *ranges,
    const float *bbox,
    const float *tri,
    const int64_t *tri_idx,
    const float tol,
    int *out,
    bool do_out
){

    int64_t stack[BVH_STACK_SIZE]; stack[0] = 0; int stack_size = 1;
    int node = 0, depth = 0;

    int cnt = 0;
    do {

        const int left_index = nodes[3 * node + 0];
        const int right_index = nodes[3 * node + 1];

        bool traverseL = false;
        if (left_index >= 0 && depth < BVH_MAX_DEPTH){
            const float3 bbox_min_left = make_float3(bbox[6 * left_index + 0], bbox[6 * left_index + 1], bbox[6 * left_index + 2]);
            const float3 bbox_max_left = make_float3(bbox[6 * left_index + 3], bbox[6 * left_index + 4], bbox[6 * left_index + 5]);
            traverseL = tri_bbox_intersect(v0, v1, v2, bbox_min_left, bbox_max_left);
        }

        bool traverseR = false;
        if(right_index >= 0 && depth < BVH_MAX_DEPTH){
            const float3 bbox_min_right = make_float3(bbox[6 * right_index + 0], bbox[6 * right_index + 1], bbox[6 * right_index + 2]);
            const float3 bbox_max_right = make_float3(bbox[6 * right_index + 3], bbox[6 * right_index + 4], bbox[6 * right_index + 5]);
            traverseR = tri_bbox_intersect(v0, v1, v2, bbox_min_right, bbox_max_right);
        }


        if ((left_index < 0 && right_index < 0) || (depth == BVH_MAX_DEPTH)) {
            const int left_bound = ranges[2 * node + 0], right_bound = ranges[2 * node + 1];
            for(int k = left_bound; k <= right_bound; k++){

                int tri_gidx = (int) tri_idx[k];
                if (tri_gidx == gidx) continue; // do not count with itself

                const float* curr_triangle = tri + 9 * k;
                const float3 u0 = make_float3(curr_triangle[3 * 0 + 0], curr_triangle[3 * 0 + 1], curr_triangle[3 * 0 + 2]);
                const float3 u1 = make_float3(curr_triangle[3 * 1 + 0], curr_triangle[3 * 1 + 1], curr_triangle[3 * 1 + 2]);
                const float3 u2 = make_float3(curr_triangle[3 * 2 + 0], curr_triangle[3 * 2 + 1], curr_triangle[3 * 2 + 2]);
                if (!tri_tri_intersect(v0, v1, v2, u0, u1, u2, tol)) continue;

                if (do_out) {
                    out[cnt * 2 + 0] = gidx;
                    out[cnt * 2 + 1] = tri_gidx;
                }
                cnt++;
            }
        }

        if (!traverseL && !traverseR){
            int64_t item = stack[--stack_size];
            node = item >> 8; depth = item & 0xFFLL;
        }
        else {
            node = traverseL ? left_index : right_index;
            depth++;
            if(traverseL && traverseR && stack_size < BVH_STACK_SIZE) {
                stack[stack_size++] = (static_cast<int64_t>(right_index) << 8) | static_cast<int64_t>(depth);
            }
        }

    } while(stack_size > 0);

    return cnt;
}


__global__ void count_intersections(
    const int n_tri,
    const float* tri,
    const int* nodes,
    const int* ranges,
    const float* bbox,
    const int64_t* tri_idx,
    const float* bvh_tri,
    const float tol,
    int* cnt
){

    int gidx = global_idx();
    if (gidx >= n_tri) return;

    const float3 v0 = make_float3(tri[gidx * 9 + 0 * 3 + 0], tri[gidx * 9 + 0 * 3 + 1], tri[gidx * 9 + 0 * 3 + 2]);
    const float3 v1 = make_float3(tri[gidx * 9 + 1 * 3 + 0], tri[gidx * 9 + 1 * 3 + 1], tri[gidx * 9 + 1 * 3 + 2]);
    const float3 v2 = make_float3(tri[gidx * 9 + 2 * 3 + 0], tri[gidx * 9 + 2 * 3 + 1], tri[gidx * 9 + 2 * 3 + 2]);


    cnt[gidx] = tri_intersects(
        v0, v1, v2, gidx,
        nodes, ranges, bbox, bvh_tri, tri_idx, tol,
        cnt + gidx, false
    );
}

__global__ void write_intersections(
    const int n_intersected,
    const int* intersected,
    const int n_tri,
    const float* tri,
    const int* nodes,
    const int* ranges,
    const float* bbox,
    const int64_t* tri_idx,
    const float* bvh_tri,
    const float tol,
    const int* cnt,
    int *intersections
){

    int thread_idx = global_idx();
    if (thread_idx >= n_intersected) return;

    int gidx = intersected[thread_idx];
    int start = gidx > 0 ? cnt[gidx - 1] : (int) 0;

    const float3 v0 = make_float3(tri[gidx * 9 + 0 * 3 + 0], tri[gidx * 9 + 0 * 3 + 1], tri[gidx * 9 + 0 * 3 + 2]);
    const float3 v1 = make_float3(tri[gidx * 9 + 1 * 3 + 0], tri[gidx * 9 + 1 * 3 + 1], tri[gidx * 9 + 1 * 3 + 2]);
    const float3 v2 = make_float3(tri[gidx * 9 + 2 * 3 + 0], tri[gidx * 9 + 2 * 3 + 1], tri[gidx * 9 + 2 * 3 + 2]);

    tri_intersects(
        v0, v1, v2, gidx,
        nodes, ranges, bbox, bvh_tri, tri_idx, tol,
        intersections + start * 2, true
    );

}

__global__ void build_verts_faces(
    const int n_faces,
    const int* faces,
    const int* vrange,
    int *vatomic,
    int *vfaces
) {
    const int gidx = global_idx();
    if (gidx >= n_faces) return;

    for(size_t k = 0; k < 3; k++){
        int vidx = faces[gidx * 3 + k];
        int start = vidx > 0 ? vrange[vidx - 1] : 0;
        int target_idx = atomicAdd(vatomic + vidx, 1);
        vfaces[start + target_idx] = gidx;
    }

}

__global__ void check_intersections(
    const int n_intersections,
    const int* intersections,
    const int* faces,
    const int *vrange,
    const int *vfaces,
    bool *ivalid
){

    const int gidx = global_idx();
    if (gidx >= n_intersections) return;

    int f0 = intersections[2 * gidx + 0], f1 = intersections[2 * gidx + 1];

    bool valid = f0 < f1;
    for(size_t k = 0; k < 3 && valid; k++){
        int vidx = faces[f0 * 3 + k];
        int start = vidx > 0 ? vrange[vidx - 1] : 0;
        int end = vrange[vidx];
        for(int j = start; j < end && valid; j++){
            valid &= vfaces[j] != f1;
        }
    }

    ivalid[gidx] = valid;
}

torch::Tensor cuda_mesh_intersection(torch::Tensor v, torch::Tensor f, float tol){

    const int threads = 128;
    int n_verts = v.size(0), n_faces = f.size(0);

    auto tri = v.index({f});
    bvh tree = build_tri_bvh(tri);
    auto sorted_tri = tri.index({tree.orig_idx});

    auto cnt = torch::zeros({n_faces}, torch::TensorOptions().dtype(torch::kInt32).device(tri.device()));
    count_intersections<<<(n_faces + threads - 1) / threads, threads>>>(
        n_faces,
        tri.data_ptr<float>(),
        tree.nodes.data_ptr<int>(),
        tree.ranges.data_ptr<int>(),
        tree.bbox.data_ptr<float>(),
        tree.orig_idx.data_ptr<int64_t>(),
        sorted_tri.data_ptr<float>(),
        tol,
        cnt.data_ptr<int>()
    );

    auto intersected = torch::arange(0, n_faces, torch::TensorOptions().dtype(torch::kInt32).device(tri.device())).index({cnt.greater(0)});
    int n_intersected = intersected.size(0);

    cnt.cumsum_(0);
    int n_intersections = cnt.index({-1}).item<int>();
    auto intersections = torch::empty({n_intersections, 2}, torch::TensorOptions().dtype(torch::kInt32).device(tri.device()));

    if (n_intersected > 0 && n_intersections > 0){
        write_intersections<<<(n_intersected + threads - 1) / threads, threads>>>(
            n_intersected,
            intersected.data_ptr<int>(),
            n_faces,
            tri.data_ptr<float>(),
            tree.nodes.data_ptr<int>(),
            tree.ranges.data_ptr<int>(),
            tree.bbox.data_ptr<float>(),
            tree.orig_idx.data_ptr<int64_t>(),
            sorted_tri.data_ptr<float>(),
            tol,
            cnt.data_ptr<int>(),
            intersections.data_ptr<int>()
        );

        auto vrange = torch::zeros({n_verts}, torch::TensorOptions().dtype(torch::kInt32).device(v.device()));
        vrange.scatter_add_(0 /*dim*/, f.view({-1}).toType(torch::kLong), torch::ones_like(f.view({-1})));
        vrange.cumsum_(0);

        auto vf = torch::empty({vrange.index({-1}).item<int>()}, torch::TensorOptions().dtype(torch::kInt32).device(v.device()));
        auto vatomic = torch::zeros({n_verts}, torch::TensorOptions().dtype(torch::kInt32).device(v.device()));
        build_verts_faces<<<(n_faces + threads - 1) / threads, threads>>>(
            n_faces,
            f.data_ptr<int>(),
            vrange.data_ptr<int>(),
            vatomic.data_ptr<int>(),
            vf.data_ptr<int>()
        );

        auto ivalid = torch::empty({n_intersections}, torch::TensorOptions().dtype(torch::kBool).device(v.device()));
        check_intersections<<<(n_intersections + threads - 1) / threads, threads>>>(
            n_intersections,
            intersections.data_ptr<int>(),
            f.data_ptr<int>(),
            vrange.data_ptr<int>(),
            vf.data_ptr<int>(),
            ivalid.data_ptr<bool>()
        );
        intersections = intersections.index({ivalid});
    }

    return intersections;
}


__global__ void fill_voxels_from_points(
    const int n,
    const float *pcd,
    const int grid_res,
    bool *out
){

    const int gidx = global_idx();
    if (gidx >= n) return;

    const float3 p = make_float3(pcd[gidx * 3 + 0], pcd[gidx * 3 + 1], pcd[gidx * 3 + 2]);
    const int x = clamp(floorf(p.x * grid_res), 0.0f, (float) (grid_res - 1));
    const int y = clamp(floorf(p.y * grid_res), 0.0f, (float) (grid_res - 1));
    const int z = clamp(floorf(p.z * grid_res), 0.0f, (float) (grid_res - 1));

    const int voxel_idx = x * grid_res * grid_res + y * grid_res + z; // maximum 1024 ^ 3
    out[voxel_idx] = true;
}

torch::Tensor cuda_voxels_from_points(torch::Tensor pcd, int grid_res){

    const int threads = 512;
    const int n = pcd.size(0);

    auto out = torch::zeros({grid_res, grid_res, grid_res}, torch::TensorOptions().dtype(torch::kBool).device(pcd.device()));
    fill_voxels_from_points<<<(n + threads - 1) / threads, threads>>>(
        n,
        pcd.data_ptr<float>(),
        grid_res,
        out.data_ptr<bool>()
    );

    return out;
}

__global__ void fill_voxels_from_triangles(
    const int n,
    const float *tri,
    const int grid_res,
    bool *out
){

    const int gidx = global_idx();
    if (gidx >= n) return;

    const float3 tri0 = make_float3(tri[gidx * 3 * 3 + 0 * 3 + 0], tri[gidx * 3 * 3 + 0 * 3 + 1], tri[gidx * 3 * 3 + 0 * 3 + 2]);
    const float3 tri1 = make_float3(tri[gidx * 3 * 3 + 1 * 3 + 0], tri[gidx * 3 * 3 + 1 * 3 + 1], tri[gidx * 3 * 3 + 1 * 3 + 2]);
    const float3 tri2 = make_float3(tri[gidx * 3 * 3 + 2 * 3 + 0], tri[gidx * 3 * 3 + 2 * 3 + 1], tri[gidx * 3 * 3 + 2 * 3 + 2]);

    const float3 tri_min = fminf(tri0, fminf(tri1, tri2)) - 1e-6f;
    const int xmin = clamp(floorf(tri_min.x * grid_res), 0.0f, (float) (grid_res - 1));
    const int ymin = clamp(floorf(tri_min.y * grid_res), 0.0f, (float) (grid_res - 1));
    const int zmin = clamp(floorf(tri_min.z * grid_res), 0.0f, (float) (grid_res - 1));

    const float3 tri_max = fmaxf(tri0, fmaxf(tri1, tri2)) + 1e-6f;
    const int xmax = clamp(floorf(tri_max.x * grid_res), 0.0f, (float) (grid_res - 1));
    const int ymax = clamp(floorf(tri_max.y * grid_res), 0.0f, (float) (grid_res - 1));
    const int zmax = clamp(floorf(tri_max.z * grid_res), 0.0f, (float) (grid_res - 1));

    for(int x = xmin; x < xmax + 1; x++){
        for(int y = ymin; y < ymax + 1; y++){
            for (int z = zmin; z < zmax + 1; z++){

                const float voxel_size = 1.0f / (float) grid_res;
                const float3 bb_min = make_float3(((float) x) * voxel_size, ((float) y) * voxel_size, ((float) z) * voxel_size) - 1e-6f;
                const float3 bb_max = make_float3(((float) x + 1.0f) * voxel_size, ((float) y + 1.0f) * voxel_size, ((float) z + 1.0f) * voxel_size) + 1e-6f;

                if(tri_bbox_intersect(tri0, tri1, tri2, bb_min, bb_max)){
                    int voxel_idx = x * grid_res * grid_res + y * grid_res + z;
                    out[voxel_idx] = true;
                }

            }
        }
    }

}


torch::Tensor cuda_voxels_from_triangles(torch::Tensor tri, int grid_res){
    const int threads = 512;
    const int n = tri.size(0);

    auto out = torch::zeros({grid_res, grid_res, grid_res}, torch::TensorOptions().dtype(torch::kBool).device(tri.device()));
    fill_voxels_from_triangles<<<(n + threads - 1) / threads, threads>>>(
        n,
        tri.data_ptr<float>(),
        grid_res,
        out.data_ptr<bool>()
    );

    return out;
}

template<bool optimized, int DILATE>
__global__ void dilate_coords_kernel(
    const int n,
    const int batch_size,
    const int4* __restrict__ coords,
    const int dil,
    bool* __restrict__ vox,
    const int grid_res
) {

    const int gidx = global_idx();
    if (gidx >= n) return;

    const int4 coord = coords[gidx];

    const int bidx = coord.x;
    const int x = coord.y;
    const int y = coord.z;
    const int z = coord.w;

    if (optimized) {
        #pragma unroll
        for(int i = -DILATE; i < DILATE + 1; i++) {
            int xx = x + i;
            if (xx < 0 || xx >= grid_res) continue;
            #pragma unroll
            for(int j = -DILATE; j < DILATE + 1; j++){
                int yy = y + j;
                if (yy < 0 || yy >= grid_res) continue;
                #pragma unroll
                for (int k = -DILATE; k < DILATE + 1; k++){
                    int zz = z + k;
                    if (zz < 0 || zz >= grid_res) continue;
                    int vox_idx = ((bidx * grid_res + xx) * grid_res + yy) * grid_res + zz;
                    vox[vox_idx] = true;
                }
            }
        }
    } else {
        for(int i = -dil; i < dil + 1; i++) {
            int xx = x + i;
            if (xx < 0 || xx >= grid_res) continue;
            for(int j = -dil; j < dil + 1; j++){
                int yy = y + j;
                if (yy < 0 || yy >= grid_res) continue;
                for (int k = -dil; k < dil + 1; k++){
                    int zz = z + k;
                    if (zz < 0 || zz >= grid_res) continue;
                    int vox_idx = ((bidx * grid_res + xx) * grid_res + yy) * grid_res + zz;
                    vox[vox_idx] = true;
                }
            }
        }
    }


}

torch::Tensor cuda_dilate_coords(torch::Tensor coords, int batch_size, int dilate, int grid_res){

    const int threads = 512;
    const int n = coords.size(0);

    torch::Tensor vox = torch::zeros({batch_size, grid_res, grid_res, grid_res}, torch::TensorOptions().dtype(torch::kBool).device(coords.device()));

    if (dilate == 1){
        dilate_coords_kernel<true, 1><<<(n + threads - 1) / threads, threads>>>(
            n,
            batch_size,
            reinterpret_cast<int4*>(coords.data_ptr<int>()),
            dilate,
            vox.data_ptr<bool>(),
            grid_res
        );
    } else if (dilate == 2){
        dilate_coords_kernel<true, 2><<<(n + threads - 1) / threads, threads>>>(
            n,
            batch_size,
            reinterpret_cast<int4*>(coords.data_ptr<int>()),
            dilate,
            vox.data_ptr<bool>(),
            grid_res
        );
    } else if (dilate == 3){
        dilate_coords_kernel<true, 3><<<(n + threads - 1) / threads, threads>>>(
            n,
            batch_size,
            reinterpret_cast<int4*>(coords.data_ptr<int>()),\
            dilate,
            vox.data_ptr<bool>(),
            grid_res
        );
    } else if (dilate == 4){
        dilate_coords_kernel<true, 4><<<(n + threads - 1) / threads, threads>>>(
            n,
            batch_size,
            reinterpret_cast<int4*>(coords.data_ptr<int>()),
            dilate,
            vox.data_ptr<bool>(),
            grid_res
        );
    } else if (dilate == 5){
        dilate_coords_kernel<true, 5><<<(n + threads - 1) / threads, threads>>>(
            n,
            batch_size,
            reinterpret_cast<int4*>(coords.data_ptr<int>()),
            dilate,
            vox.data_ptr<bool>(),
            grid_res
        );
    } else {
        dilate_coords_kernel<false, 0><<<(n + threads - 1) / threads, threads>>>(
            n,
            batch_size,
            reinterpret_cast<int4*>(coords.data_ptr<int>()),
            dilate,
            vox.data_ptr<bool>(),
            grid_res
        );
    }


    return vox;
}

__global__ void active_voxels_kernel(
    const int N,
    const float *sdf,
    const int GRID_X,
    const int GRID_Y,
    const int GRID_Z,
    bool *vox_mask
){

    int gidx = global_idx();
    if (gidx >= N) return;

    int z = gidx % GRID_Z;
    int y = (gidx / GRID_Z) % GRID_Y;
    int x = (gidx / (GRID_Y * GRID_Z)) % GRID_X;

    int pos = 0;
    for (int vox_idx = 0; vox_idx < 8; vox_idx++){

        int xx = x + static_cast<int>((vox_idx & 1) > 0);
        int yy = y + static_cast<int>((vox_idx & 2) > 0);
        int zz = z + static_cast<int>((vox_idx & 4) > 0);
        int target_idx = (xx * (GRID_Y + 1) + yy) * (GRID_Z + 1) + zz;

        pos += static_cast<int> (sdf[target_idx] > 0);
    }

    vox_mask[gidx] = (pos % 8 != 0);
}

torch::Tensor cuda_active_voxels(torch::Tensor sdf){
    constexpr int threads = 512;
    auto device = sdf.device();

    int GRID_X = sdf.size(0) - 1, GRID_Y = sdf.size(1) - 1, GRID_Z = sdf.size(2) - 1;
    int N = GRID_X * GRID_Y * GRID_Z;

    auto vox_mask = torch::empty({GRID_X, GRID_Y, GRID_Z}, torch::TensorOptions().device(device).dtype(torch::kBool));
    active_voxels_kernel<<<(N + threads - 1) / threads, threads>>>(
        N,
        sdf.data_ptr<float>(),
        GRID_X,
        GRID_Y,
        GRID_Z,
        vox_mask.data_ptr<bool>()
    );

    return torch::nonzero(vox_mask);
}

__global__ void near_voxels_kernel(
    const int N,
    const float thr,
    const float *dist,
    const int GRID_X,
    const int GRID_Y,
    const int GRID_Z,
    bool *vox_mask
){

    int gidx = global_idx();
    if (gidx >= N) return;

    int z = gidx % GRID_Z;
    int y = (gidx / GRID_Z) % GRID_Y;
    int x = (gidx / (GRID_Y * GRID_Z)) % GRID_X;

    int near_cnt = 0;
    for (int vox_idx = 0; vox_idx < 8; vox_idx++){

        int xx = x + static_cast<int>((vox_idx & 1) > 0);
        int yy = y + static_cast<int>((vox_idx & 2) > 0);
        int zz = z + static_cast<int>((vox_idx & 4) > 0);
        int target_idx = (xx * (GRID_Y + 1) + yy) * (GRID_Z + 1) + zz;

        near_cnt += static_cast<int> (fabsf(dist[target_idx]) <= thr);
    }

    vox_mask[gidx] = near_cnt > 0;
}

torch::Tensor cuda_near_voxels(torch::Tensor dist, const float thr){

    constexpr int threads = 512;
    auto device = dist.device();

    int GRID_X = dist.size(0) - 1, GRID_Y = dist.size(1) - 1, GRID_Z = dist.size(2) - 1;
    int N = GRID_X * GRID_Y * GRID_Z;

    auto vox_mask = torch::empty({GRID_X, GRID_Y, GRID_Z}, torch::TensorOptions().device(device).dtype(torch::kBool));
    near_voxels_kernel<<<(N + threads - 1) / threads, threads>>>(
        N,
        thr,
        dist.data_ptr<float>(),
        GRID_X,
        GRID_Y,
        GRID_Z,
        vox_mask.data_ptr<bool>()
    );

    return torch::nonzero(vox_mask);
}

__device__ __forceinline__ uint64_t get_voxel_corner_code(const int4& coord, const int grid_res)
{
    return ((static_cast<uint64_t>(coord.x) * grid_res + coord.y) * grid_res + coord.z) * grid_res + coord.w;
}

__global__ void fill_vox_kernel(
    const int N_CUBES,
    const int grid_res,
    const int4* __restrict__ coords,
    bool* __restrict__ vox
){

    const int gidx = global_idx();
    if (gidx >= N_CUBES) return;

    vox[get_voxel_corner_code(coords[gidx], grid_res)] = true;

}

__global__ void find_corner_points_kernel(
    const int N_POINTS,
    const int grid_res,
    const int4* __restrict__ coords,
    const bool* __restrict__ vox,
    uint8_t* __restrict__ corner_bits,
    bool* __restrict__ corner_mask
){

    const int gidx = global_idx();
    if (gidx >= N_POINTS) return;

    const int4 coord = coords[gidx];
    const int bidx = coord.x;
    const int x = coord.y, y = coord.z, z = coord.w;

    uint8_t msk = 0;
    #pragma unroll
    for (int shift = 0; shift < 8; shift++)
    {
        const int zz = z - ((shift & (1 << 0)) >> 0);
        const int yy = y - ((shift & (1 << 1)) >> 1);
        const int xx = x - ((shift & (1 << 2)) >> 2);
        if ((xx >= 0 && yy >= 0 && zz >= 0) && (xx < grid_res && yy < grid_res && zz < grid_res))
        {
            const int4 dcoord = make_int4(bidx, xx, yy, zz);
            if (vox[get_voxel_corner_code(dcoord, grid_res)])
            {
                msk |= static_cast<uint8_t>(1u << shift);
            }
        }
    }

    corner_bits[gidx] = msk;
    corner_mask[gidx] = msk < 0xFF;

}

__global__ void find_corner_neighbours_kernel(
    const int N_CORNER, const int grid_res,
    const int4* __restrict__ coords,
    const uint8_t* __restrict__ corner_bits,
    int3* __restrict__ neighbours
) {

    const int gidx = global_idx();
    if (gidx >= N_CORNER) return;

    const int4 coord = coords[gidx];
    const int bidx = coord.x;
    const int x = coord.y, y = coord.z, z = coord.w;

    const int4 lower_coord = make_int4(bidx, max(x - 1, 0), max(y - 1, 0), max(z - 1, 0));
    const uint64_t lower_code = get_voxel_corner_code(lower_coord, grid_res + 1);
    int l = 0, r = N_CORNER;
    while (r - l > 1)
    {
        const int m = (l + r) >> 1;
        if (get_voxel_corner_code(coords[m], grid_res + 1) <= lower_code) l = m;
        else r = m;
    }
    const int lower_bound = l;

    const int4 upper_coord = make_int4(bidx, min(x + 1, grid_res), min(y + 1, grid_res), min(z + 1, grid_res));
    const uint64_t upper_code = get_voxel_corner_code(upper_coord, grid_res + 1);
    l = lower_bound, r = N_CORNER;
    while (r - l > 1)
    {
        const int m = (l + r) >> 1;
        if (get_voxel_corner_code(coords[m], grid_res + 1) <= upper_code) l = m;
        else r = m;
    }
    const int upper_bound = l + 1;


    bool free_edge[6];
    #pragma unroll
    for (size_t k = 0; k < 6; k++) free_edge[k] = false;

    const uint8_t cube_mask = corner_bits[gidx];
    #pragma unroll
    for (int shift = 0; shift < 8; shift++)
    {
        if ((cube_mask & (1u << shift)) == 0)
        {
            if (shift & 1) free_edge[0 * 2 + 0] = true; // -z
            else free_edge[0 * 2 + 1] = true;

            if (shift & 2) free_edge[1 * 2 + 0] = true; // -y
            else free_edge[1 * 2 + 1] = true;

            if (shift & 4) free_edge[2 * 2 + 0] = true; // -x
            else free_edge[2 * 2 + 1] = true;
        }
    }

    int ids[6];
    #pragma unroll
    for (int k = 0; k < 6; k++)
    {
        ids[k] = -1;

        int zyx[3] = {z, y, x};
        zyx[k / 2] += 2 * (k % 2) - 1;

        if (free_edge[k] && (zyx[0] >= 0 && zyx[1] >= 0 && zyx[2] >= 0) && (zyx[0] <= grid_res && zyx[1] <= grid_res && zyx[2] <= grid_res))
        {
            const int4 ncoord = make_int4(bidx, zyx[2], zyx[1], zyx[0]);
            const uint64_t ncode = get_voxel_corner_code(ncoord, grid_res + 1);
            l = lower_bound, r = upper_bound;
            while (r - l > 1)
            {
                const int m = (l + r) >> 1;
                if (get_voxel_corner_code(coords[m], grid_res + 1) <= ncode) l = m;
                else r = m;
            }
            if (get_voxel_corner_code(coords[l], grid_res + 1) == ncode) ids[k] = l;
        }
    }

    neighbours[gidx * 2 + 0] = make_int3(ids[0], ids[1], ids[2]);
    neighbours[gidx * 2 + 1] = make_int3(ids[3], ids[4], ids[5]);
}


__global__ void propagate_corner_kernel(
    const int N_CORNER,
    const int3* __restrict__ neighbours,
    int* __restrict__ clabel
) {

    const int gidx = global_idx();
    if (gidx >= N_CORNER) return;

    const int3 ids0 = neighbours[gidx * 2 + 0];
    const int3 ids1 = neighbours[gidx * 2 + 1];
    const int ids[6] = {ids0.x, ids0.y, ids0.z, ids1.x, ids1.y, ids1.z};

    #pragma unroll
    for (int k = 0; k < 6; k++)
    {
        if (ids[k] >= 0)
        {
            dsu_union(gidx, ids[k], clabel);
        }
    }

}


torch::Tensor cuda_find_corner_points(torch::Tensor vidx, torch::Tensor cube_idx, int batch_size, int grid_res)
{
    constexpr int threads = 256;
    auto device = vidx.device();

    const int N_CUBES = cube_idx.size(0);
    auto vox = torch::zeros({batch_size, grid_res, grid_res, grid_res}, torch::TensorOptions().dtype(torch::kBool).device(device));
    fill_vox_kernel<<<(N_CUBES + threads - 1) / threads, threads>>>(
        N_CUBES, grid_res,
        reinterpret_cast<int4*>(cube_idx.data_ptr<int>()), vox.data_ptr<bool>()
    );

    const int N_POINTS = vidx.size(0);
    auto corner_bits = torch::empty({N_POINTS}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    auto corner_mask = torch::empty({N_POINTS}, torch::TensorOptions().dtype(torch::kBool).device(device));
    find_corner_points_kernel<<<(N_POINTS + threads - 1) / threads, threads>>>(
        N_POINTS, grid_res,
        reinterpret_cast<int4*>(vidx.data_ptr<int>()), vox.data_ptr<bool>(),
        corner_bits.data_ptr<uint8_t>(), corner_mask.data_ptr<bool>()
    );
    corner_bits = corner_bits.index({corner_mask});

    auto corner_idx = vidx.index({corner_mask});
    const int N_CORNER = corner_idx.size(0);
    auto corner_neighbours = torch::empty({N_CORNER, 6}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    find_corner_neighbours_kernel<<<(N_CORNER + threads - 1) / threads, threads>>>(
        N_CORNER, grid_res,
        reinterpret_cast<int4*>(corner_idx.data_ptr<int>()),
        corner_bits.data_ptr<uint8_t>(),
        reinterpret_cast<int3*>(corner_neighbours.data_ptr<int>())
    );

    auto clabel = torch::arange(N_CORNER, torch::TensorOptions().dtype(torch::kInt32).device(device));
    propagate_corner_kernel<<<(N_CORNER + threads - 1) / threads, threads>>>(
        N_CORNER,
        reinterpret_cast<int3*>(corner_neighbours.data_ptr<int>()),
        clabel.data_ptr<int>()
    );
    dsu_compact_kernel<<<(N_CORNER + threads - 1) / threads, threads>>>(
        N_CORNER, clabel.data_ptr<int>()
    );

    auto out_labels = torch::full({N_POINTS}, -1, torch::TensorOptions().dtype(torch::kInt32).device(device));
    out_labels.index_put_({corner_mask}, clabel);

    return out_labels;
}

__global__ void init_exterior_voxels_kernel(
    const int N_ITEMS, const int grid_res,
    const bool* __restrict__ occ,
    int32_t* __restrict__ label,
    uint8_t* __restrict__ cmask
) {

    const int gidx = global_idx();
    if (gidx == N_ITEMS)
    {
        label[gidx] = N_ITEMS;
    }

    if (gidx >= N_ITEMS) return;

    const int N_VOXELS = grid_res * grid_res * grid_res;
    const int z = gidx % grid_res;
    const int y = (gidx / grid_res) % grid_res;
    const int x = (gidx / (grid_res * grid_res)) % grid_res;

    const bool is_occupied = occ[gidx];
    if (is_occupied)
    {
        label[gidx] = gidx;
    }
    else if (x > 0 && y > 0 && z > 0 && x < grid_res - 1 && y < grid_res - 1 && z < grid_res - 1)
    {
        label[gidx] = gidx;
    }
    else
    {
        label[gidx] = N_ITEMS;
    }

    uint8_t msk = 0u;
    #pragma unroll
    for (int shift = 0; shift < 6; shift++)
    {
        int zyx[3] = {z, y, x};
        zyx[shift / 2] += 2 * (shift % 2) - 1;
        if (!is_occupied && (zyx[0] >= 0 && zyx[1] >= 0 && zyx[2] >= 0) && (zyx[0] < grid_res && zyx[1] < grid_res && zyx[2] < grid_res))
        {
            const int4 cube_coord = make_int4(gidx / N_VOXELS, zyx[2], zyx[1], zyx[0]);
            if (!occ[get_voxel_corner_code(cube_coord, grid_res)]) msk |= (1u << shift);
        }
    }
    cmask[gidx] = msk;

}

__global__ void merge_exterior_voxels_kernel(
    const int N_ITEMS, const int grid_res,
    const uint8_t* __restrict__ cube_mask,
    int32_t* __restrict__ label
) {

    const int gidx = global_idx();
    if (gidx >= N_ITEMS) return;

    const int N_VOXELS = grid_res * grid_res * grid_res;
    const int z = gidx % grid_res;
    const int y = (gidx / grid_res) % grid_res;
    const int x = (gidx / (grid_res * grid_res)) % grid_res;

    const uint8_t cmask = cube_mask[gidx];
    #pragma unroll
    for (int shift = 0; shift < 6; shift++)
    {
        if (cmask & (1u << shift))
        {
            int zyx[3] = {z, y, x};
            zyx[shift / 2] += 2 * (shift % 2) - 1;
            const int4 cube_coord = make_int4(gidx / N_VOXELS, zyx[2], zyx[1], zyx[0]);
            dsu_union(gidx, static_cast<int32_t>(get_voxel_corner_code(cube_coord, grid_res)), label);
        }
    }
}

__global__ void mark_exterior_voxels_kernel(
    const int N_ITEMS,
    int32_t* __restrict__ label,
    bool* __restrict__ ext
) {

    const int gidx = global_idx();
    if (gidx >= N_ITEMS) return;

    // merge tree
    dsu_find(gidx, label);
    if (label[gidx] == N_ITEMS)
    {
        ext[gidx] = true;
    } else
    {
        ext[gidx] = false;
    }

}

torch::Tensor cuda_exterior_voxels(torch::Tensor cube_idx, int batch_size, int grid_res)
{
    constexpr int threads = 256;
    auto device = cube_idx.device();

    const int N_CUBES = cube_idx.size(0);
    auto occ = torch::zeros({batch_size, grid_res, grid_res, grid_res}, torch::TensorOptions().dtype(torch::kBool).device(device));
    fill_vox_kernel<<<(N_CUBES + threads - 1) / threads, threads>>>(
        N_CUBES, grid_res,
        reinterpret_cast<int4*>(cube_idx.data_ptr<int>()), occ.data_ptr<bool>()
    );

    const int N_ITEMS = occ.numel();
    auto label = torch::empty({N_ITEMS + 1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto cube_mask = torch::empty({batch_size, grid_res, grid_res, grid_res}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    init_exterior_voxels_kernel<<<(N_ITEMS + 1 + threads - 1) / threads, threads>>>(
        N_ITEMS, grid_res,
        occ.data_ptr<bool>(),
        label.data_ptr<int>(),
        cube_mask.data_ptr<uint8_t>()
    );
    merge_exterior_voxels_kernel<<<(N_ITEMS + threads - 1) / threads, threads>>>(
        N_ITEMS, grid_res,
        cube_mask.data_ptr<uint8_t>(),
        label.data_ptr<int>()
    );

    auto ext = torch::zeros({batch_size, grid_res, grid_res, grid_res}, torch::TensorOptions().dtype(torch::kBool).device(device));
    mark_exterior_voxels_kernel<<<(N_ITEMS + threads - 1) / threads, threads>>>(
        N_ITEMS, label.data_ptr<int>(),
        ext.data_ptr<bool>()
    );

    return ext;
}


__global__ void mark_exterior_corners_kernel(
    const int N_CORNERS, const int grid_res,
    const int4* __restrict__ coords,
    const bool* __restrict__ ext_vox,
    bool* __restrict__ ext_corner
) {
    const int gidx = global_idx();
    if (gidx >= N_CORNERS) return;

    const int4 coord = coords[gidx];
    const int bidx = coord.x;
    const int x = coord.y, y = coord.z, z = coord.w;

    bool is_ext = false;
    #pragma unroll
    for (int shift = 0; shift < 8; shift++)
    {
        int zz = z - ((shift & (1 << 0)) >> 0);
        int yy = y - ((shift & (1 << 1)) >> 1);
        int xx = x - ((shift & (1 << 2)) >> 2);
        if ((xx >= 0 && yy >= 0 && zz >= 0) && (xx < grid_res && yy < grid_res && zz < grid_res))
        {
            const int4 cube_coord = make_int4(bidx, xx, yy, zz);
            if (ext_vox[get_voxel_corner_code(cube_coord, grid_res)]) is_ext = true;
        } else is_ext = true;
    }

    ext_corner[gidx] = is_ext;
}

torch::Tensor cuda_mark_corners_from_exterior_voxels(torch::Tensor vidx, torch::Tensor ext_vox, int grid_res)
{

    constexpr int threads = 256;
    auto device = vidx.device();

    const int N_CORNERS = vidx.size(0);
    auto ext_corner = torch::empty({N_CORNERS}, torch::TensorOptions().dtype(torch::kBool).device(device));
    mark_exterior_corners_kernel<<<(N_CORNERS + threads - 1) / threads, threads>>>(
        N_CORNERS, grid_res,
        reinterpret_cast<int4*>(vidx.data_ptr<int>()),
        ext_vox.data_ptr<bool>(),
        ext_corner.data_ptr<bool>()
    );
    return ext_corner;
}

__device__ __forceinline__ uint32_t expand_bits_3d(uint32_t v){
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__ int32_t morton_3d(int32_t x, int32_t y, int32_t z){
    return static_cast<int32_t>((expand_bits_3d((uint32_t) x) << 2) | (expand_bits_3d((uint32_t) y) << 1) | expand_bits_3d((uint32_t) z));
}

__global__ void vert2cube_kernel(
    const int N_VERTS,
    const int grid_res,
    const int3* __restrict__ v_idx,
    int32_t* __restrict__ cube_mask
) {
    const int gidx = global_idx();
    if (gidx >= N_VERTS) return;

    const int3 v = v_idx[gidx];

    #pragma unroll
    for (int k = 0; k < 8; k++){
        const int zz = v.z - ((k & (1 << 0)) >> 0);
        const int yy = v.y - ((k & (1 << 1)) >> 1);
        const int xx = v.x - ((k & (1 << 2)) >> 2);
        if (xx >= 0 && yy >= 0 && zz >= 0 && xx < grid_res && yy < grid_res && zz < grid_res){
            atomicAdd(cube_mask + (xx * grid_res + yy) * grid_res + zz, 1);
        }
    }
}

torch::Tensor cuda_vert2cube(torch::Tensor v_idx, int grid_res){

    constexpr int threads = 256;
    auto device = v_idx.device();

    const int N_VERTS = v_idx.size(0);
    auto cube_mask = torch::zeros({grid_res, grid_res, grid_res}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    vert2cube_kernel<<<(N_VERTS + threads - 1) / threads, threads>>>(
        N_VERTS, grid_res,
        reinterpret_cast<int3*>(v_idx.data_ptr<int>()),
        cube_mask.data_ptr<int32_t>()
    );
    
    return cube_mask.eq(static_cast<int32_t>(8)).nonzero().to(torch::kInt16);
}

std::vector<torch::Tensor> cuda_vertex_faces(torch::Tensor faces)
{


    std::tuple<torch::Tensor, torch::Tensor> _vorder = torch::sort(faces.view({-1}), true, 0, false);
    auto vids = std::get<0>(_vorder);
    auto vorder = std::get<1>(_vorder);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _vcnt = torch::unique_consecutive(vids, false, true);
    auto vcnt = std::get<2>(_vcnt);
    auto voff = vcnt.cumsum(0, torch::kInt32);

    auto face_idx = torch::arange(faces.size(0), torch::TensorOptions().dtype(torch::kInt32).device(faces.device()));
    face_idx = face_idx.repeat_interleave(3);
    face_idx = face_idx.index({vorder});

    auto vfaces = torch::stack({vids, face_idx}, -1);

    return {vfaces, vcnt,voff};
}

__global__ void add_first_item_kernel(
    const int32_t* __restrict__ item2item,
    const int32_t* __restrict__ off,
    const int N_ITEMS,
    int2* __restrict__ pairs
)
{
    const int gidx = global_idx();
    if (gidx >= N_ITEMS) return;

    int start = gidx > 0 ? off[gidx - 1] : static_cast<int>(0);
    int end = off[gidx];

    for (int index = start; index < end; index++)
    {
        pairs[index] = make_int2(gidx, item2item[index]);
    }
}

torch::Tensor cuda_add_first_item(torch::Tensor item2item, torch::Tensor off)
{

    constexpr int threads = 256;
    const int N_ITEMS = off.size(0);

    auto pairs = torch::empty({item2item.size(0), 2}, item2item.options());
    add_first_item_kernel<<<(N_ITEMS + threads - 1) / threads, threads>>>(
        item2item.data_ptr<int32_t>(),
        off.data_ptr<int32_t>(),
        N_ITEMS,
        reinterpret_cast<int2*>(pairs.data_ptr<int32_t>())
    );
    return pairs;
}

__global__ void face2face_codes_kernel(
    const int N_FACES,
    const int3* __restrict__ faces,
    const int32_t* __restrict__ foff,
    const int2* __restrict__ vfaces,
    const int32_t* __restrict__ voff,
    int64_t* __restrict__ face_codes
) {

    const int gidx = global_idx();
    if (gidx >= N_FACES) return;

    const int3 face = faces[gidx];
    int vids[3] = {face.x, face.y, face.z};

    int fstart = gidx > 0 ? foff[gidx - 1] : static_cast<int32_t>(0);

    #pragma unroll
    for (int k = 0; k < 3; ++k)
    {
        int vstart = vids[k] > 0 ? voff[vids[k] - 1] : static_cast<int>(0);
        int vend = voff[vids[k]];
        for (int index = vstart; index < vend; ++index)
        {
            face_codes[fstart++] = (static_cast<int64_t>(gidx) << 32) | static_cast<int64_t>(vfaces[index].y);
        }
    }
}

__global__ void face2face_bounds_kernel(
    const int N_FACES,
    const int N_CODES,
    const int64_t* __restrict__ face_codes,
    int32_t* __restrict__ face_bounds
)
{
    const int gidx = global_idx();
    if (gidx >= N_FACES) return;

    const int64_t max_code = (static_cast<int64_t>(gidx) << 32) | ((1LL << 32) - 1);
    int l = 0, r = N_CODES;
    while (r - l > 1)
    {
        const int mid = (l + r) >> 1;
        if (face_codes[mid] <= max_code) l = mid;
        else r = mid;
    }
    face_bounds[gidx] = l + 1;
}

__global__ void off2cnt_kernel(
    const int N,
    const int32_t* __restrict__ off,
    int32_t* __restrict__ cnt
)
{
    const int gidx = global_idx();
    if (gidx >= N) return;
    const int lbound = gidx > 0 ? off[gidx - 1] : static_cast<int>(0);
    const int rbound = off[gidx];
    cnt[gidx] = rbound - lbound;
}

__global__ void uncouple_face_code_kernel(
    const int N_PAIRS,
    const int64_t* __restrict__ face_codes,
    int32_t* __restrict__ face2face
)
{
    const int gidx = global_idx();
    if (gidx >= N_PAIRS) return;

    const int64_t face_code = face_codes[gidx];

    // const int face0 = static_cast<int32_t>(face_code >> 32);
    const int face1 = static_cast<int32_t>(face_code & ((1LL << 32) - 1));
    face2face[gidx] = face1;

}

std::vector<torch::Tensor> cuda_topo_neighbours(torch::Tensor faces)
{


    auto _vf = cuda_vertex_faces(faces);
    auto vfaces = _vf[0], vcnt = _vf[1], voff = _vf[2];

    constexpr int threads = 256;
    auto device = faces.device();

    auto fcnt = vcnt.index({faces}).sum(-1);
    fcnt = fcnt.cumsum(0, torch::kInt32);
    auto face_codes = torch::empty({
        fcnt.index({-1}).item<int32_t>()
    }, torch::TensorOptions().dtype(torch::kInt64).device(device));

    const int N_FACES = faces.size(0);
    face2face_codes_kernel<<<(N_FACES + threads - 1) / threads, threads>>>(
        N_FACES,
        reinterpret_cast<int3*>(faces.data_ptr<int32_t>()),
        fcnt.data_ptr<int32_t>(),
        reinterpret_cast<int2*>(vfaces.data_ptr<int32_t>()),
        voff.data_ptr<int32_t>(),
        face_codes.data_ptr<int64_t>()
    );

    face_codes = std::get<0>(torch::_unique(face_codes));

    auto foff = torch::empty({N_FACES}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    face2face_bounds_kernel<<<(N_FACES + threads - 1) / threads, threads>>>(
        N_FACES,
        static_cast<int32_t>(face_codes.size(0)),
        face_codes.data_ptr<int64_t>(),
        foff.data_ptr<int32_t>()
    );

    // fcnt = torch::empty({N_FACES}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    // off2cnt_kernel<<<(N_FACES + threads - 1) / threads, threads>>>(
    //     N_FACES,
    //     foff.data_ptr<int32_t>(),
    //     fcnt.data_ptr<int32_t>()
    // );

    const int N_FACE_PAIRS = face_codes.size(0);
    auto face2face = torch::empty({N_FACE_PAIRS}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    uncouple_face_code_kernel<<<(N_FACE_PAIRS + threads - 1) / threads, threads>>>(
        N_FACE_PAIRS,
        face_codes.data_ptr<int64_t>(),
        face2face.data_ptr<int32_t>()
    );

    return {face2face, foff};
}


__device__ __forceinline__ int bvh_tri_ball(
    const float3 &p, const float radius,
    const int3* __restrict__ nodes, const int2* __restrict__ ranges,
    const float3* __restrict__ bbox, const float3* __restrict__ tri,
    const int64_t* __restrict__ orig_idx, int32_t* __restrict__ face2face
){

    constexpr int TREE_DEPTH = 32;
    constexpr int TREE_STACK = TREE_DEPTH * 2;

    int fcnt = 0;
    int64_t stack[TREE_STACK]; stack[0] = 0; int stack_size = 1;

    int node = 0, depth = 0;
    do {

        const int3 childs = nodes[node];

        bool traverseL = false;
        if (childs.x >= 0 && depth < TREE_DEPTH){
            const float3 bbox_min_left = bbox[childs.x * 2 + 0];
            const float3 bbox_max_left = bbox[childs.x * 2 + 1];
            traverseL = bbox_dist(p, bbox_min_left, bbox_max_left) <= radius;
        }

        bool traverseR = false;
        if(childs.y >= 0 && depth < TREE_DEPTH){
            const float3 bbox_min_right = bbox[childs.y * 2 + 0];
            const float3 bbox_max_right = bbox[childs.y * 2 + 1];
            traverseR = bbox_dist(p, bbox_min_right, bbox_max_right) <= radius;
        }

        if ((childs.x < 0 && childs.y < 0) || (depth == TREE_DEPTH)) {
            // LEAF
            const int2 bounds = ranges[node];
            for(int tri_idx = bounds.x; tri_idx <= bounds.y; tri_idx++){
                const float3* curr_tri = tri + tri_idx * 3;
                const float3 pp = (curr_tri[0] + curr_tri[1] + curr_tri[2]) / 3.0f;

                if (length(p - pp) <= radius)
                {
                    if (face2face) face2face[fcnt] = static_cast<int>(orig_idx[tri_idx]);
                    fcnt++;
                }
            }
        }


        if (!traverseL && !traverseR){
            int64_t item = stack[--stack_size];
            node = item >> 8; depth = item & 0xFFLL;
        }
        else if(traverseL && traverseR && stack_size < TREE_STACK){
            node = traverseL ? childs.x : childs.y;
            stack[stack_size++] = (static_cast<int64_t>(traverseL ? childs.y : childs.x) << 8) | (static_cast<int64_t>(++depth));
        }
        else {
            node = traverseL ? childs.x : childs.y;
            depth++;
        }

    } while(stack_size > 0);


    return fcnt;
}

__global__ void count_radius_faces_kernel(
    const int N_FACES,
    const float3* __restrict__ tri,
    const int3* __restrict__ nodes,
    const int2* __restrict__ ranges,
    const float3* __restrict__ bbox,
    const float3* __restrict__ sorted_tri,
    const float radius,
    int* __restrict__ fcnt
)
{
    const int gidx = global_idx();
    if (gidx >= N_FACES) return;

    const float3* curr_tri = &tri[gidx * 3];
    const float3 p = (curr_tri[0] + curr_tri[1] + curr_tri[2]) / 3.0f;
    fcnt[gidx] = bvh_tri_ball(p, radius, nodes, ranges, bbox, sorted_tri, nullptr, nullptr);

}

__global__ void put_radius_faces_kernel(
    const int N_FACES,
    const float3* __restrict__ tri,
    const int3* __restrict__ nodes,
    const int2* __restrict__ ranges,
    const float3* __restrict__ bbox,
    const float3* __restrict__ sorted_tri,
    const int64_t* __restrict__ orig_idx,
    const float radius,
    const int32_t* __restrict__ foff,
    int32_t* __restrict__ face2face
) {
    const int gidx = global_idx();
    if (gidx >= N_FACES) return;

    const float3* curr_tri = &tri[gidx * 3];
    const float3 p = (curr_tri[0] + curr_tri[1] + curr_tri[2]) / 3.0f;

    int fstart = gidx > 0 ? foff[gidx - 1] : static_cast<int32_t>(0);
    bvh_tri_ball(p, radius, nodes, ranges, bbox, sorted_tri, orig_idx, face2face + fstart);

}


std::vector<torch::Tensor> cuda_geo_neighbours(torch::Tensor verts, torch::Tensor faces, float radius)
{

    auto tri = verts.index({faces}).div_(2).add_(0.5);
    bvh tree = build_tri_bvh(tri);
    auto sorted_tri = tri.index({tree.orig_idx});

    auto device = tri.device();
    constexpr int threads = 128;

    const int N_FACES = tri.size(0);
    auto fcnt = torch::empty({N_FACES}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    count_radius_faces_kernel<<<(N_FACES + threads - 1) / threads, threads>>>(
        N_FACES,
        reinterpret_cast<float3*>(tri.data_ptr<float>()),
        reinterpret_cast<int3*>(tree.nodes.data_ptr<int32_t>()),
        reinterpret_cast<int2*>(tree.ranges.data_ptr<int32_t>()),
        reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
        reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
        radius / 2,
        fcnt.data_ptr<int32_t>()
    );

    auto foff = fcnt.cumsum(0, torch::kInt32);
    const int N_FACE_PAIRS = foff.index({-1}).item<int32_t>();
    auto face2face = torch::empty({N_FACE_PAIRS}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    put_radius_faces_kernel<<<(N_FACES + threads - 1) / threads, threads>>>(
        N_FACES,
        reinterpret_cast<float3*>(tri.data_ptr<float>()),
        reinterpret_cast<int3*>(tree.nodes.data_ptr<int32_t>()),
        reinterpret_cast<int2*>(tree.ranges.data_ptr<int32_t>()),
        reinterpret_cast<float3*>(tree.bbox.data_ptr<float>()),
        reinterpret_cast<float3*>(sorted_tri.data_ptr<float>()),
        tree.orig_idx.data_ptr<int64_t>(),
        radius / 2,
        foff.data_ptr<int32_t>(),
        face2face.data_ptr<int32_t>()
    );
    // TODO select only faces connected with main face
    return {face2face, foff};
}

__global__ void face_normals_kernel(
    const int N_FACES,
    const int3* __restrict__ faces,
    const float3* __restrict__ verts,
    float* __restrict__ areas,
    float3* __restrict__ normals
)
{
    const int gidx = global_idx();
    if (gidx >= N_FACES) return;

    const int3 face = faces[gidx];
    const float3 v0 = verts[face.x], v1 = verts[face.y], v2 = verts[face.z];
    const float3 nrm = cross(v1 - v0, v2 - v0);

    areas[gidx] = length(nrm) * 0.5f;
    normals[gidx] = nrm / static_cast<float>(sqrt(static_cast<double>(nrm.x) * nrm.x + static_cast<double>(nrm.y) * nrm.y + static_cast<double>(nrm.z) * nrm.z) + 1e-12f);

}

__device__ __forceinline__ bool is_adjacent(int* __restrict__ fids0, int* __restrict__ fids1)
{
    #pragma unroll
    for (size_t k = 0; k < 3; ++k)
    {
        #pragma unroll
        for (size_t kk = 0; kk < 3; ++kk)
        {
            if (
                min(fids0[k], fids0[(k + 1) % 3]) == min(fids1[kk], fids1[(kk + 1) % 3]) &&
                max(fids0[k], fids0[(k + 1) % 3]) == max(fids1[kk], fids1[(kk + 1) % 3])
            ) return true;
        }
    }
    return false;
}

__global__ void face_patch_variance_kernel(
    const int N_FACES,
    const int3* __restrict__ faces,
    const int32_t* __restrict__ foff,
    const int32_t* __restrict__ face2face,
    const float* __restrict__ face_areas,
    const float3* __restrict__ face_normals,
    float3* __restrict__ patch_normals,
    float* __restrict__ patch_variance
) {
    const int gidx = global_idx();
    if (gidx >= N_FACES) return;

    const int fstart = gidx > 0 ? foff[gidx - 1] : 0;
    const int fend = foff[gidx];

    // average normal and patch area
    double3 nrm = make_double3(0.0, 0.0, 0.0);
    double area = 1e-12;
    for (int i = fstart; i < fend; ++i)
    {
        const int fidx0 = face2face[i];
        const float area0 = face_areas[fidx0];
        area += static_cast<double>(area0);
        const float3 nrm0 = face_normals[fidx0];
        nrm = make_double3(
            nrm.x + static_cast<double>(area0) * nrm0.x,
            nrm.y + static_cast<double>(area0) * nrm0.y,
            nrm.z + static_cast<double>(area0) * nrm0.z
        );
    }
    nrm = nrm / area;
    nrm = nrm / (sqrt(dot(nrm, nrm)) + 1e-12);
    patch_normals[gidx] = make_float3(static_cast<float>(nrm.x), static_cast<float>(nrm.y), static_cast<float>(nrm.z));

    double nrm_var = 0.0f;
    for (int i = fstart; i < fend; ++i)
    {

        const int fidx0 = face2face[i];
        // const int3 face0 = faces[fidx0];
        const float area0 = face_areas[fidx0];
        const float3 nrm0 = face_normals[fidx0];

        nrm_var += acos(
            fmin(fmax(nrm.x * nrm0.x + nrm.y * nrm0.y + nrm.z * nrm0.z, -1.0 + 1e-8), 1.0 - 1e-8)
        ) * (area0 / area);

//         for(int j = i + 1; j < fend; ++j){
//             const int fidx1 = face2face[j];
//             const float area1 = face_areas[fidx1];
//             const float3 nrm1 = face_normals[fidx1];
//             nrm_var += acosf(fminf(fmaxf(dot(nrm0, nrm1), -1.0 + 1e-8f), 1.0 - 1e-8f));
//         }
    }

    float variance = static_cast<float>(nrm_var);
    if (sqrt(dot(nrm, nrm)) < 0.5) variance = 1e6f; // invalid normal
    patch_variance[gidx] = variance;

}

__global__ void assign_face_patches_kernel(
    const int N_FACES,
    const int32_t* __restrict__ foff,
    const int32_t* __restrict__ face2face,
    const float3* __restrict__ patch_normals,
    const float* __restrict__ patch_variance,
    int32_t* __restrict__ face_lock,
    float* __restrict__ min_variance,
    float3* __restrict__ guidance_normals
)
{
    const int gidx = global_idx();
    if (gidx >= N_FACES) return;

    const int fstart = gidx > 0 ? foff[gidx - 1] : 0;
    const int fend = foff[gidx];

    const float3 nrm = patch_normals[gidx];
    const float var = patch_variance[gidx];

    for (int index = fstart; index < fend; ++index) {
        const int fidx = face2face[index];
        if (__int_as_float(atomicMin(reinterpret_cast<int*>(&min_variance[fidx]), __float_as_int(var))) > var)
        {
            while (atomicCAS(&face_lock[fidx], 0, 1) != 0) {}
            guidance_normals[fidx] = nrm;
            __threadfence();
            atomicExch(&face_lock[fidx], 0);
        }
    }
}

std::vector<torch::Tensor> cuda_guidance_normals(
    torch::Tensor verts, torch::Tensor faces,
    torch::Tensor face2face, torch::Tensor foff
) {


    constexpr int threads = 256;
    auto device = verts.device();
    const int N_FACES = faces.size(0);

    auto face_areas = torch::empty({N_FACES}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto face_normals = torch::empty({N_FACES, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    face_normals_kernel<<<(N_FACES + threads - 1) / threads, threads>>>(
        N_FACES,
        reinterpret_cast<int3*>(faces.data_ptr<int32_t>()),
        reinterpret_cast<float3*>(verts.data_ptr<float>()),
        face_areas.data_ptr<float>(),
        reinterpret_cast<float3*>(face_normals.data_ptr<float>())
    );

    auto patch_normals = torch::empty({N_FACES, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto patch_variance = torch::empty({N_FACES}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    face_patch_variance_kernel<<<(N_FACES + threads - 1) / threads, threads>>>(
        N_FACES,
        reinterpret_cast<int3*>(faces.data_ptr<int32_t>()),
        foff.data_ptr<int32_t>(),
        face2face.data_ptr<int32_t>(),
        face_areas.data_ptr<float>(),
        reinterpret_cast<float3*>(face_normals.data_ptr<float>()),
        reinterpret_cast<float3*>(patch_normals.data_ptr<float>()),
        patch_variance.data_ptr<float>()
    );

    auto face_lock = torch::zeros({N_FACES}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto min_variance = patch_variance.clone();
    auto guidance_normals = patch_normals.clone();
    assign_face_patches_kernel<<<(N_FACES + threads - 1) / threads, threads>>>(
        N_FACES,
        foff.data_ptr<int32_t>(),
        face2face.data_ptr<int32_t>(),
        reinterpret_cast<float3*>(patch_normals.data_ptr<float>()),
        patch_variance.data_ptr<float>(),
        face_lock.data_ptr<int32_t>(),
        min_variance.data_ptr<float>(),
        reinterpret_cast<float3*>(guidance_normals.data_ptr<float>())
    );

    return { guidance_normals};
}

__global__ void neighbour_faces_ori_score_kernel(
    const int2* __restrict__ face2face,
    const int32_t N_PAIRS,
    const int3* __restrict__ faces,
    const float3* __restrict__ face_normals,
    const float3* __restrict__ verts,
    const float sigma_dist, const float sigma_nrm,
    float3* __restrict__ scores
)
{
    const int gidx = global_idx();
    if (gidx >= N_PAIRS) return;

    const int2 pair = face2face[gidx];

    const int3 face0 = faces[pair.x], face1 = faces[pair.y];

    const float3 v0 = verts[face1.x], v1 = verts[face1.y], v2 = verts[face1.z];
    const float area = length(cross(v1 - v0, v2 - v0)) * 0.5f;

    const float3 c0 = (verts[face0.x] + verts[face0.y] + verts[face0.z]) / 3.0f;
    const float3 c1 = (verts[face1.x] + verts[face1.y] + verts[face1.z]) / 3.0f;
    const float kdist = expf(-powf(length(c0 - c1) / sigma_dist, 2.0f) * 0.5f);

    const float3 nrm0 = face_normals[pair.x];
    const float3 nrm1 = face_normals[pair.y];
    const float angle = acosf(fminf(fmaxf(dot(nrm0, nrm1), -1.0f + 1e-6f), 1.0f - 1e-6f));
    const float kangle = expf(-powf(angle / sigma_nrm, 2.0f) * 0.5f);

    // add face with itself? TODO
    if (pair.x != pair.y)
    {
        scores[gidx] = make_float3(area, kdist, kangle);
    } else
    {
        scores[gidx] = make_float3(0.0f, 0.0f, 0.0f);
    }

}

__global__ void normalize_neighbour_scores_kernel(
    const int32_t* __restrict__ foff,
    const int N_ITEMS,
    const float3* __restrict__ pair_scores,
    float* __restrict__ scores
){
    const int gidx = global_idx();
    if (gidx >= N_ITEMS) return;

    int fstart = gidx > 0 ? foff[gidx - 1] : static_cast<int>(0);
    int fend = foff[gidx];

    double denom = 1e-12;
    for (int index = fstart; index < fend; index++)
    {
        const float3 pair_score = pair_scores[index];
        denom += static_cast<double>(pair_score.x) * pair_score.y * pair_score.z;
    }

    for (int index = fstart; index < fend; index++)
    {
        const float3 pair_score = pair_scores[index];
        scores[index] = static_cast<float>(static_cast<double>(pair_score.x) * pair_score.y * pair_score.z / denom);
    }

}

torch::Tensor cuda_neighbour_faces_ori_scores(
    torch::Tensor verts, torch::Tensor faces,
    torch::Tensor face_normals, torch::Tensor face2face, torch::Tensor foff,
    float sigma_dist, float sigma_nrm
) {

    constexpr int threads = 256;
    auto device = face_normals.device();

    const int N_PAIRS = face2face.size(0);

    auto pair_scores = torch::empty({N_PAIRS, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    neighbour_faces_ori_score_kernel<<<(N_PAIRS + threads - 1) / threads, threads>>>(
        reinterpret_cast<int2*>(face2face.data_ptr<int32_t>()),
        N_PAIRS,
        reinterpret_cast<int3*>(faces.data_ptr<int32_t>()),
        reinterpret_cast<float3*>(face_normals.data_ptr<float>()),
        reinterpret_cast<float3*>(verts.data_ptr<float>()),
        sigma_dist,
        sigma_nrm,
        reinterpret_cast<float3*>(pair_scores.data_ptr<float>())
    );

    const int N_ITEMS = faces.size(0);
    auto scores = torch::empty({N_PAIRS}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    normalize_neighbour_scores_kernel<<<(N_ITEMS + threads - 1) / threads, threads>>>(
        foff.data_ptr<int32_t>(),
        N_ITEMS,
        reinterpret_cast<float3*>(pair_scores.data_ptr<float>()),
        scores.data_ptr<float>()
    );

    return scores;
}
