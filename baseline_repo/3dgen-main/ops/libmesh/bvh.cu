#include <cutil_math.h>
#include <bvh.h>
using namespace torch::indexing;


__device__ uint32_t expandBits(uint32_t v){
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ uint64_t morton3D(float x, float y, float z, int idx){
    x = clamp(rintf(x * 1024.0f), 0.0f, 1023.0f);
    y = clamp(rintf(y * 1024.0f), 0.0f, 1023.0f);
    z = clamp(rintf(z * 1024.0f), 0.0f, 1023.0f);
    uint32_t xx = expandBits((uint32_t) x);
    uint32_t yy = expandBits((uint32_t) y);
    uint32_t zz = expandBits((uint32_t) z);
    uint64_t geo_code = (zz << 2) | (yy << 1) | xx;
    uint64_t idx_code = (uint32_t) idx;
    return (geo_code << 32) | idx_code;
}

__global__ void morton_code_kernel(
    const int bsz,
    const int n_points,
    const float3* __restrict__ pcd,
    int64_t* __restrict__ codes
){

    const int thread_idx = global_idx();
    if (thread_idx >= bsz * n_points)
        return;

    const float3 pnt = pcd[thread_idx];
    codes[thread_idx] = (int64_t) morton3D(pnt.x, pnt.y, pnt.z, thread_idx % n_points);
}

__global__ void leaf_kernel(
    const int bsz,
    const int n_points,
    const float3* __restrict__ sorted_pcd,
    int3* __restrict__ leaf_nodes,
    int2* __restrict__ leaf_ranges
) {
    const int thread_idx = global_idx();
    if (thread_idx >= bsz * n_points)
        return;

    int batch_idx = thread_idx / n_points;
    int offset = batch_idx * n_points;
    int pidx = thread_idx - offset;
    leaf_ranges[thread_idx] = make_int2(pidx, pidx);

    leaf_nodes[thread_idx] = make_int3(-1, -1, -1);
}


__device__ int common_prefix(uint64_t a, uint64_t b){
    return __clzll((int64_t) (a ^ b));
}

__global__ void build_tree_kernel(
    const int bsz,
    const int n_points,
    const int64_t* __restrict__ codes,
    int* nodes
){

    const int gidx = global_idx();
    if (gidx >= bsz * (n_points - 1))
        return;

    const int batch_idx = gidx / (n_points - 1);
    const int pidx = gidx % (n_points - 1);
    const int offset = batch_idx * n_points;
    const uint64_t curr_code = (uint64_t) codes[offset + pidx];

    int sign, left_bound, right_bound;
    if(pidx == 0){
        sign = 1, left_bound = 0, right_bound = n_points - 1;
    } else {
        const uint64_t prev_code = (uint64_t) codes[offset + pidx - 1];
        const uint64_t next_code = (uint64_t) codes[offset + pidx + 1];
        sign = common_prefix(curr_code, next_code) - common_prefix(curr_code, prev_code);
        sign = max(min(sign, 1), -1);

        const int min_prefix = sign < 0 ? common_prefix(curr_code, next_code) : common_prefix(prev_code, curr_code);
        if (sign > 0){
            int l = pidx, r = n_points;
            while(r - l > 1){
                int m = (l + r) / 2;
                if(common_prefix(curr_code, (uint64_t) codes[offset + m]) > min_prefix) l = m;
                else r = m;
            }
            left_bound = pidx, right_bound = l;
        } else if (sign < 0){
            int l = -1, r = pidx;
            while(r - l > 1){
                int m = (l + r) / 2;
                if(common_prefix(curr_code, (uint64_t) codes[offset + m]) > min_prefix) r = m;
                else l = m;
            }
            left_bound = r, right_bound = pidx;
        }
    }

    const uint64_t left_code = sign > 0 ? curr_code : (uint64_t) codes[offset + left_bound];
    const uint64_t right_code = sign > 0 ? (uint64_t) codes[offset + right_bound] : curr_code;
    const int node_prefix = common_prefix(left_code, right_code);

    int l = left_bound, r = right_bound + 1;
    while(r - l > 1){
        int m = (l + r) / 2;
        if(common_prefix(left_code, (uint64_t) codes[offset + m]) > node_prefix) l = m;
        else r = m;
    }
    const int split_index = l;

    // for leaf nodes shifts by number of internal nodes
    const int inode = batch_idx * (2 * n_points - 1) + pidx;
    const int common_left_index = split_index, common_right_index = split_index + 1;
    const int left_index = (common_left_index - left_bound > 0) ? common_left_index : (n_points - 1 + common_left_index);
    nodes[inode * 3 + 0] = left_index;
    const int right_index = (right_bound - common_right_index > 0) ? common_right_index : (n_points - 1 + common_right_index);
    nodes[inode * 3 + 1] = right_index;

    if (pidx == 0) nodes[inode * 3 + 2] = -1;
    nodes[(batch_idx * (2 * n_points - 1) + left_index) * 3 + 2] = pidx;
    nodes[(batch_idx * (2 * n_points - 1) + right_index) * 3 + 2] = pidx;

}


__global__ void internal_node_kernel(
    const int bsz,
    const int n_points,
    const int* __restrict__ nodes,
    int* __restrict__ ranges,
    float3* __restrict__ bbox,
    int* __restrict__ flags
){

    const int gidx = global_idx();
    if (gidx >= bsz * n_points)
        return;
    const int batch_idx = gidx / n_points;
    const int pidx = gidx % n_points;
    const int offset = batch_idx * (2 * n_points - 1);

    int curr_node = nodes[(offset + n_points - 1 + pidx) * 3 + 2];
    while(curr_node != -1){

        if(atomicCAS(flags + batch_idx * (n_points - 1) + curr_node, 0, 1) == 0) return;

        const int left_index = nodes[(offset + curr_node) * 3 + 0];
        const int right_index = nodes[(offset + curr_node) * 3 + 1];

        ranges[(offset + curr_node) * 2 + 0] = ranges[(offset + left_index) * 2 + 0];
        ranges[(offset + curr_node) * 2 + 1] = ranges[(offset + right_index) * 2 + 1];

        bbox[(offset + curr_node) * 2 + 0] = fminf(bbox[(offset + left_index) * 2 + 0], bbox[(offset + right_index) * 2 + 0]);
        bbox[(offset + curr_node) * 2 + 1] = fmaxf(bbox[(offset + left_index) * 2 + 1], bbox[(offset + right_index) * 2 + 1]);

        __threadfence();
        curr_node = nodes[(offset + curr_node) * 3 + 2];
    }

}

bvh build_bvh(torch::Tensor pcd, torch::Tensor pcd_bbox){
    const int bsz = pcd.size(0), n_points = pcd.size(1);

    int threads = 128;
    auto codes = torch::empty(
        {bsz, n_points},
        torch::TensorOptions().dtype(torch::kInt64).device(pcd.device())
    );
    morton_code_kernel<<<(bsz * n_points + threads - 1) / threads, threads>>>(
        bsz,
        n_points,
        reinterpret_cast<float3*>(pcd.data_ptr<float>()),
        codes.data_ptr<int64_t>()
    );

    std::tuple<torch::Tensor, torch::Tensor> sort_out = torch::sort(
        codes,
        1, // dim=1
        false // descending=False
    );
    codes = std::get<0>(sort_out);
    auto sorted_idx = std::get<1>(sort_out);
    auto sorted_pcd = pcd.index({
        torch::arange(0, bsz, torch::TensorOptions().dtype(torch::kInt64).device(codes.device())).view({bsz, 1}),
        sorted_idx
    });
    if (pcd_bbox.size(2) == 3){ // b n 3
        pcd_bbox = pcd_bbox.repeat({1, 1, 2});
    }
    auto sorted_bbox = pcd_bbox.index({
        torch::arange(0, bsz, torch::TensorOptions().dtype(torch::kInt64).device(codes.device())).view({bsz, 1}),
        sorted_idx
    });

    // FIND DUPLICATE CODES AND RANGES FOR THEM WITH TORCH
    auto leaf_nodes = torch::empty(
        {bsz, n_points, 3},
        torch::TensorOptions().dtype(torch::kInt32).device(codes.device())
    );
    auto leaf_ranges = torch::empty(
        {bsz, n_points, 2},
        torch::TensorOptions().dtype(torch::kInt32).device(codes.device())
    );
    auto leaf_bbox = sorted_bbox.reshape({bsz, n_points, 6});
    leaf_kernel<<<(bsz * n_points + threads - 1) / threads, threads>>>(
        bsz,
        n_points,
        reinterpret_cast<float3*>(sorted_pcd.data_ptr<float>()),
        reinterpret_cast<int3*>(leaf_nodes.data_ptr<int>()),
        reinterpret_cast<int2*>(leaf_ranges.data_ptr<int>())
    );

    // BUILD BVH CONCURRENTLY FOR EACH INTERNAL NODE
    auto internal_nodes = torch::empty(
        {bsz, n_points - 1, 3},
        torch::TensorOptions().dtype(torch::kInt32).device(codes.device())
    );
    auto internal_ranges = torch::empty(
        {bsz, n_points - 1, 2},
        torch::TensorOptions().dtype(torch::kInt32).device(codes.device())
    );
    auto internal_bbox = torch::empty(
        {bsz, n_points - 1, 6},
        torch::TensorOptions().dtype(torch::kFloat32).device(codes.device())
    );
    auto nodes = torch::cat({internal_nodes, leaf_nodes}, 1 /*dim*/);
    auto ranges = torch::cat({internal_ranges, leaf_ranges}, 1 /*dim*/);
    auto bbox = torch::cat({internal_bbox, leaf_bbox}, 1 /*dim*/);

    build_tree_kernel<<<(bsz * (n_points - 1) + threads - 1) / threads, threads>>>(
        bsz,
        n_points,
        codes.data_ptr<int64_t>(),
        nodes.data_ptr<int>()
    );

    // BOTTOM UP BOUNDING BOX CALCULATION FOR TREE
    auto flags = torch::zeros(
        {bsz, n_points - 1},
        torch::TensorOptions().dtype(torch::kInt32).device(codes.device())
    );
    internal_node_kernel<<<(bsz * n_points + threads - 1) / threads, threads>>>(
        bsz,
        n_points,
        nodes.data_ptr<int>(),
        ranges.data_ptr<int>(),
        reinterpret_cast<float3*>(bbox.data_ptr<float>()),
        flags.data_ptr<int>()
    );

    return {nodes, ranges, bbox, sorted_pcd, sorted_idx};
}

bvh build_tri_bvh(torch::Tensor tri){

    auto bbox_min = std::get<0>(tri.min(1 /*dim*/)), bbox_max = std::get<0>(tri.max(1 /*dim*/));
    auto bbox = torch::cat({bbox_min, bbox_max}, 1);
    auto pcd = bbox.view({-1, 2, 3}).mean(1/*dim*/);

    bvh tree = build_bvh(pcd.view({1, -1, 3}), bbox.view({1, -1, 6}));

    return {
        tree.nodes.view({-1, 3}),
        tree.ranges.view({-1, 2}),
        tree.bbox.view({-1, 6}),
        tree.points.view({-1, 3}),
        tree.orig_idx.view({-1})
    };
}