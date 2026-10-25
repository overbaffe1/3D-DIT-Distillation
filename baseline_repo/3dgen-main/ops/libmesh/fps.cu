

#include <cutil_math.h>
#include <meshi.h>
using namespace torch::indexing;


#define STACK_SIZE 128
struct far_query {
    float dist;
    int64_t idx;
};

__device__ far_query point_far_query(const float3 p, const int *nodes, const int *ranges, const float *bbox, const float *pcd){
    float4 mindist = make_float4(1e6, 0.0f, 0.0f, 1.0f);

    int stack[STACK_SIZE];
    stack[0] = -1;
    int stack_size = 1;

    int node = 0;
    do {

        const int left_index = nodes[3 * node + 0];
        const int right_index = nodes[3 * node + 1];

        bool traverseL = false;
        float left_dist = 1e6;
        if (left_index >= 0){
            const float3 bbox_min_left = make_float3(bbox[6 * left_index + 0], bbox[6 * left_index + 1], bbox[6 * left_index + 2]);
            const float3 bbox_max_left = make_float3(bbox[6 * left_index + 3], bbox[6 * left_index + 4], bbox[6 * left_index + 5]);
            left_dist = bbox_dist(p, bbox_min_left, bbox_max_left);
            traverseL = left_dist < mindist.x;
        }

        bool traverseR = false;
        float right_dist = 1e6;
        if(right_index >= 0){
            const float3 bbox_min_right = make_float3(bbox[6 * right_index + 0], bbox[6 * right_index + 1], bbox[6 * right_index + 2]);
            const float3 bbox_max_right = make_float3(bbox[6 * right_index + 3], bbox[6 * right_index + 4], bbox[6 * right_index + 5]);
            right_dist = bbox_dist(p, bbox_min_right, bbox_max_right);
            traverseR = right_dist < mindist.x;
        }

        if (left_index < 0 && right_index < 0) {
            // LEAF
            const int left_bound = ranges[2 * node + 0], right_bound = ranges[2 * node + 1];
            for(int tri_idx = left_bound; tri_idx <= right_bound; tri_idx++){
                const float* curr_triangle = tri + 9 * tri_idx;
                const float3 v0 = make_float3(curr_triangle[3 * 0 + 0], curr_triangle[3 * 0 + 1], curr_triangle[3 * 0 + 2]);
                const float3 v1 = make_float3(curr_triangle[3 * 1 + 0], curr_triangle[3 * 1 + 1], curr_triangle[3 * 1 + 2]);
                const float3 v2 = make_float3(curr_triangle[3 * 2 + 0], curr_triangle[3 * 2 + 1], curr_triangle[3 * 2 + 2]);

                // FIND MIN DISTANCE
                const float4 dp = p_tri_dist(p, v0, v1, v2);
                mindist = pminp(mindist, dp);
            }
        }


        if (!traverseL && !traverseR){
            stack_size--;
            node = stack[stack_size];
        }
        else if(traverseL && traverseR && stack_size < STACK_SIZE){
            node = (left_dist < right_dist) ? left_index : right_index;
            stack[stack_size] = (left_dist < right_dist) ? right_index : left_index;
            stack_size++;
        }
        else {
            node = traverseL ? left_index : right_index;
        }

    } while(node != -1);

    return mindist;
}


#define THREADS 512

__global__ void tree_fps(
    const int bsz,
    const int n_points,
    const int n_fps,
    const int *nodes,
    const int *ranges,
    const float *bbox,
    const float *pcd,
    int64_t *fps_idx
) {
    const int bidx = blockIdx.x, lidx = threadIdx.x;
    const int target_offset = bidx * n_fps, pcd_offset = bidx * n_points, tree_offset = bidx * (2 * n_points - 1);
    if (lidx == 0) fps_idx[target_offset + 0] = 0;

    __shared__ float best_dist[THREADS];
    __shared__ int64_t best_dist_idx[THREADS];

    for(int ncurr = 1; ncurr < n_fps; ncurr++){

        // initialize with some default values
        __syncthreads();
        best_dist[lidx] = 0.0f;
        best_dist_idx[lidx] = 0;

        int wave_offset = 0;
        for(int waveidx = 0; waveidx < (ncurr + THREADS - 1) / THREADS; waveidx++){

            int curr_idx = wave_offset + lidx;
            if (curr_idx >= ncurr) continue;

            int point_idx = fps_idx[target_offset + curr_idx];
            float3 point = make_float3(pcd[(pcd_offset + point_idx) * 3 + 0], pcd[(pcd_offset + point_idx) * 3 + 1], pcd[(pcd_offset + point_idx) * 3 + 2]);
            far_query item = point_far_query(
                point,
                nodes + tree_offset,
                ranges + tree_offset,
                bbox + tree_offset,
                pcd + pcd_offset
            );

            if (item.dist > best_dist[lidx]){
                best_dist[lidx] = item.dist;
                best_dist_idx[lidx] = item.idx;
            }

            wave_offset += THREADS;
        }

        // aggregate group minimum point
        for (int step = THREADS / 2; step > 0; step /= 2) {
            __syncthreads();
            if (lidx < step && best_dist[lidx] < best_dist[lidx + step]) {
                best_dist[lidx] = best_dist[lidx + step];
                best_dist_idx[lidx] = best_dist_idx[lidx + step];
            }
        }
        __syncthreads();

        // write next minimum point
        if (lidx == 0) fps_idx[target_offset + ncurr] = best_dist_idx[0];
    }

}

torch::Tensor cuda_fps(torch::Tensor pcd, int n_fps) {

    bvh tree = build_bvh(pcd);

    int bsz = pcd.size(0), n_points = pcd.size(1);
    auto fps_idx = torch::empty({bsz, n_fps}, torch::TensorOptions().dtype(torch::kInt64).device(pcd.device()));

    tree_fps<<<bsz, THREADS>>>(
        bsz,
        n_points,
        n_fps,
        tree.nodes.data_ptr<int>(),
        tree.ranges.data_ptr<int>(),
        tree.bbox.data_ptr<float>(),
        tree.points.data_ptr<float>(),
        fps_idx.data_ptr<int64_t>()
    );

    return tree.sorted_idx.index({
        torch::arange(0, bsz, torch::TensorOptions().dtype(torch::kInt64).device(fps_idx.device())).view({bsz, 1}),
        fps_idx
    });
}
