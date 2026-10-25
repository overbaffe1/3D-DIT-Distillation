#ifndef LIBSPARSE_PCD_ATTN_H
#define LIBSPARSE_PCD_ATTN_H

#include <torch/extension.h>

struct attn_out {
    torch::Tensor out;
    torch::Tensor nn_idx;
    torch::Tensor nn_count;
    torch::Tensor head_max;
    torch::Tensor head_denom;
    attn_out(
        torch::Tensor out_,
        torch::Tensor nn_idx_,
        torch::Tensor nn_count_,
        torch::Tensor head_max_,
        torch::Tensor head_denom_)
        : out(out_), nn_idx(nn_idx_), nn_count(nn_count_), head_max(head_max_), head_denom(head_denom_) {}
};

std::vector<torch::Tensor> cuda_pcd_knn(
    torch::Tensor keypoints,
    torch::Tensor points,
    int K,
    float radius);

std::vector<torch::Tensor> cuda_pcd_vox_knn(
    torch::Tensor keypoints,
    torch::Tensor points,
    int grid_res,
    int dilation);

attn_out cuda_pcd_attn_fwd(
    torch::Tensor keypoints,
    torch::Tensor key_feat,
    torch::Tensor points,
    torch::Tensor point_kfeat,
    torch::Tensor point_vfeat,
    torch::Tensor point_extras,
    torch::Tensor nn_idx,
    torch::Tensor nn_count,
    torch::Tensor weights,
    torch::Tensor bias,
    int head_dim,
    int K,
    float coord_scale,
    float rel_clamp,
    int proj_share_group);

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
    int proj_share_group);

#endif
