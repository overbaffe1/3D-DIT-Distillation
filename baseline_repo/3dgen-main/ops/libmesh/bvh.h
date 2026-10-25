
#ifndef CUBVH_H
#define CUBVH_H

#include <cuda_runtime.h>
#include <torch/extension.h>

struct bvh{
    torch::Tensor nodes, ranges, bbox, points, orig_idx;
    bvh(torch::Tensor nodes, torch::Tensor ranges, torch::Tensor bbox, torch::Tensor points, torch::Tensor orig_idx): nodes(nodes), ranges(ranges), bbox(bbox), points(points), orig_idx(orig_idx) {}
};

bvh build_bvh(torch::Tensor pcd, torch::Tensor pcd_bbox);

bvh build_tri_bvh(torch::Tensor tri);

#endif