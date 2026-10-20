#include <torch/extension.h>
#include "pcd_attn.h"
namespace py = pybind11;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat, #x " must be float")
#define CHECK_INT(x) TORCH_CHECK(x.dtype() == torch::kInt32, #x " must be int32")
#define CHECK_INPUT_FLOAT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)
#define CHECK_INPUT_INT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT(x)


std::vector<torch::Tensor> conv_indices_cuda(torch::Tensor pos);

torch::Tensor vox_connected_components_cuda(torch::Tensor pos);

torch::Tensor vox_hierarchical_patches_cuda(torch::Tensor pos, torch::Tensor code, int max_patch_size);

torch::Tensor conv_fwd_cuda(torch::Tensor feat, torch::Tensor weight, torch::Tensor bias, std::vector<torch::Tensor> indices);

std::vector<torch::Tensor> conv_dw_cuda(torch::Tensor feat, torch::Tensor grad_out, std::vector<torch::Tensor> indices);

torch::Tensor dense_conv_fwd_cuda(torch::Tensor feat, torch::Tensor weight, torch::Tensor bias, std::vector<torch::Tensor> indices);

std::vector<torch::Tensor> dense_conv_dw_cuda(torch::Tensor feat, torch::Tensor grad_out, std::vector<torch::Tensor> indices);

torch::Tensor dense_conv_dx_cuda(torch::Tensor grad_out, torch::Tensor weight, std::vector<torch::Tensor> indices);

std::vector<torch::Tensor> conv_indices(
    torch::Tensor pos
) {

    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 4, "wrong coord shape");
    CHECK_INPUT_INT(pos);

    return conv_indices_cuda(pos);
}


torch::Tensor conv_fwd(
    torch::Tensor feat, // N, C
    torch::Tensor weight, // K K K C
    torch::Tensor bias, // C
    std::vector<torch::Tensor> indices
) {

    TORCH_CHECK(bias.dim() == 1, "bias must have 1 dimensions");
    TORCH_CHECK(weight.dim() == 4, "weight must have 4 dimensions");
    TORCH_CHECK(feat.dim() == 2 && feat.size(1) == weight.size(3) && feat.size(1) == bias.size(0), "wrong input shape");
    CHECK_CUDA(feat); CHECK_CONTIGUOUS(feat); CHECK_INPUT_FLOAT(weight); CHECK_INPUT_FLOAT(bias);
    for(torch::Tensor idx: indices) {
        CHECK_CUDA(idx); CHECK_CONTIGUOUS(idx);
    }

    return conv_fwd_cuda(feat, weight, bias, indices);
}

std::vector<torch::Tensor> conv_dw(
    torch::Tensor feat, // N, C
    torch::Tensor grad_out, // N, C
    std::vector<torch::Tensor> indices
) {

    TORCH_CHECK(feat.dim() == 2, "wrong input shape");
    TORCH_CHECK(grad_out.dim() == 2 && grad_out.size(0) == feat.size(0) && grad_out.size(1) == feat.size(1), "wrong grad shape");
    TORCH_CHECK(feat.dtype() == grad_out.dtype(), "feat and grad dtype must match");
    CHECK_CUDA(feat); CHECK_CONTIGUOUS(feat); CHECK_CUDA(grad_out); CHECK_CONTIGUOUS(grad_out);

    return conv_dw_cuda(feat, grad_out, indices);
}

torch::Tensor conv_dx(
    torch::Tensor grad_out, // N, C
    torch::Tensor weight, // 3,3,3,C
    std::vector<torch::Tensor> indices
) {

    TORCH_CHECK(weight.dim() == 4, "weight must have 4 dimensions");
    TORCH_CHECK(grad_out.dim() == 2 && grad_out.size(1) == weight.size(3), "wrong grad shape");
    CHECK_CUDA(grad_out); CHECK_CONTIGUOUS(grad_out); CHECK_INPUT_FLOAT(weight);
    for(torch::Tensor idx: indices) {
        CHECK_CUDA(idx); CHECK_CONTIGUOUS(idx);
    }

    weight = torch::flip(weight, {0, 1, 2}).contiguous();
    auto bias = torch::zeros({weight.size(3)}, torch::TensorOptions().device(weight.device()).dtype(weight.dtype()));

    return conv_fwd_cuda(grad_out, weight, bias, indices);
}

// TODO DEBUG
torch::Tensor dense_conv_fwd(
    torch::Tensor feat,   // N, C_IN
    torch::Tensor weight, // K,K,K,C_IN,C_OUT
    torch::Tensor bias,   // C_OUT
    std::vector<torch::Tensor> indices
) {
    TORCH_CHECK(bias.dim() == 1, "bias must have 1 dimensions");
    TORCH_CHECK(weight.dim() == 5, "weight must have 5 dimensions");
    TORCH_CHECK(feat.dim() == 2 && feat.size(1) == weight.size(3) && bias.size(0) == weight.size(4), "wrong input shape");
    CHECK_CUDA(feat); CHECK_CONTIGUOUS(feat); CHECK_INPUT_FLOAT(weight); CHECK_INPUT_FLOAT(bias);
    for(torch::Tensor idx: indices) {
        CHECK_CUDA(idx); CHECK_CONTIGUOUS(idx);
    }
    return dense_conv_fwd_cuda(feat, weight, bias, indices);
}

torch::Tensor dense_conv_dx(
    torch::Tensor grad_out, // N, C_OUT
    torch::Tensor weight,   // K,K,K,C_IN,C_OUT
    std::vector<torch::Tensor> indices
) {
    TORCH_CHECK(weight.dim() == 5, "weight must have 5 dimensions");
    TORCH_CHECK(grad_out.dim() == 2 && grad_out.size(1) == weight.size(4), "wrong grad shape");
    CHECK_CUDA(grad_out); CHECK_CONTIGUOUS(grad_out); CHECK_INPUT_FLOAT(weight);
    for(torch::Tensor idx: indices) {
        CHECK_CUDA(idx); CHECK_CONTIGUOUS(idx);
    }
    weight = torch::flip(weight, {0, 1, 2});
    weight = weight.permute({0, 1, 2, 4, 3}).contiguous();
    auto bias = torch::zeros({weight.size(4)}, torch::TensorOptions().device(weight.device()).dtype(weight.dtype()));
    return dense_conv_fwd_cuda(grad_out, weight, bias, indices);
}

std::vector<torch::Tensor> dense_conv_dw(
    torch::Tensor feat,     // N, C_IN
    torch::Tensor grad_out, // N, C_OUT
    std::vector<torch::Tensor> indices
) {
    TORCH_CHECK(feat.dim() == 2, "wrong input shape");
    TORCH_CHECK(grad_out.dim() == 2 && grad_out.size(0) == feat.size(0), "wrong grad shape");
    CHECK_CUDA(feat); CHECK_CONTIGUOUS(feat); CHECK_CUDA(grad_out); CHECK_CONTIGUOUS(grad_out);
    return dense_conv_dw_cuda(feat, grad_out, indices);
}



torch::Tensor vox_connected_components(torch::Tensor pos) {
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 4, "pos must be [N,4]");
    CHECK_INPUT_INT(pos);
    return vox_connected_components_cuda(pos);
}

torch::Tensor vox_hierarchical_patches(
    torch::Tensor pos, torch::Tensor code, int max_patch_size)
{
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 4, "pos must be [N,4]");
    TORCH_CHECK(code.dim() == 1 && code.size(0) == pos.size(0), "code must be [N]");
    TORCH_CHECK(code.dtype() == torch::kInt64, "code must be int64");
    CHECK_INPUT_INT(pos);
    CHECK_CUDA(code); CHECK_CONTIGUOUS(code);
    return vox_hierarchical_patches_cuda(pos, code, max_patch_size);
}

std::vector<torch::Tensor> pcd_knn(
    torch::Tensor keypoints,
    torch::Tensor points,
    int K, float radius
) {
    CHECK_INPUT_FLOAT(keypoints);
    CHECK_INPUT_FLOAT(points);
    TORCH_CHECK(
        (keypoints.dim() == 2 && keypoints.size(1) == 3) ||
        (keypoints.dim() == 3 && keypoints.size(2) == 3),
        "keypoints must be (M,3) or (B,M,3)"
    );
    TORCH_CHECK(
        (points.dim() == 2 && points.size(1) == 3) ||
        (points.dim() == 3 && points.size(2) == 3),
        "points must be (N,3) or (B,N,3)"
    );
    TORCH_CHECK(keypoints.dim() == points.dim(),
        "keypoints and points must both be batched or both be unbatched");
    if (keypoints.dim() == 3) {
        TORCH_CHECK(keypoints.size(0) == points.size(0),
            "keypoints and points batch size mismatch");
    }
    TORCH_CHECK(K >= 1 && K <= 1024, "K must be between 1 and 1024");
    TORCH_CHECK(radius >= 0.0 && radius <= 1.733, "radius must be between 0.0 and sqrt(3)");

    return cuda_pcd_knn(keypoints, points, K, radius);
}

std::vector<torch::Tensor> pcd_vox_knn(
    torch::Tensor keypoints,
    torch::Tensor points,
    int grid_res,
    int dilation
) {
    CHECK_INPUT_FLOAT(keypoints);
    CHECK_INPUT_FLOAT(points);
    TORCH_CHECK(
        (keypoints.dim() == 2 && keypoints.size(1) == 3) ||
        (keypoints.dim() == 3 && keypoints.size(2) == 3),
        "keypoints must be (M,3) or (B,M,3)"
    );
    TORCH_CHECK(
        (points.dim() == 2 && points.size(1) == 3) ||
        (points.dim() == 3 && points.size(2) == 3),
        "points must be (N,3) or (B,N,3)"
    );
    TORCH_CHECK(keypoints.dim() == points.dim(),
        "keypoints and points must both be batched or both be unbatched");
    if (keypoints.dim() == 3) {
        TORCH_CHECK(keypoints.size(0) == points.size(0),
            "keypoints and points batch size mismatch");
    }
    TORCH_CHECK(grid_res >= 2 && grid_res <= 4096, "grid_res must be in [2, 2048]");
    TORCH_CHECK(dilation >= 0 && dilation <= 32, "dilation must be in [0, 32]");

    return cuda_pcd_vox_knn(keypoints, points, grid_res, dilation);
}

std::vector<torch::Tensor> pcd_attn(
    torch::Tensor keypoints,
    torch::Tensor key_feat,
    torch::Tensor points,
    torch::Tensor point_kfeat,
    torch::Tensor point_vfeat,
    torch::Tensor point_extras,
    torch::Tensor weights,
    torch::Tensor bias,
    int head_dim, int K, float radius, float coord_scale, float rel_clamp, int proj_share_group,
    torch::Tensor nn_idx_in,
    torch::Tensor nn_count_in
) {
#ifdef PCD_ATTN_ONLY_HD96
    TORCH_CHECK(head_dim == 96, "This build only supports head_dim=96");
#else
    TORCH_CHECK(
        head_dim == 48 || head_dim == 72 || head_dim == 96,
        "Supported head_dim values are {48,72,96}"
    );
#endif

    const bool has_nn_idx = (nn_idx_in.defined() && nn_idx_in.numel() > 0);
    const bool has_nn_count = (nn_count_in.defined() && nn_count_in.numel() > 0);
    TORCH_CHECK(has_nn_idx == has_nn_count, "nn_idx and nn_count must be provided together or both omitted");

    if (!has_nn_idx) {
        TORCH_CHECK(K >= 1 && K <= 1024, "K must be between 1 and 1024");
        TORCH_CHECK(radius >= 0.0 && radius <= 1.733, "radius must be between 0.0 and sqrt(3)");
    }
    TORCH_CHECK(coord_scale >= 0.0, "coord_scale must be non-negative");
    TORCH_CHECK(rel_clamp >= 0.0, "rel_clamp must be non-negative");
    TORCH_CHECK(proj_share_group == 1 || proj_share_group == 4, "proj_share_group must be 1 or 4");

    CHECK_INPUT_FLOAT(keypoints);
    CHECK_INPUT_FLOAT(points);
    CHECK_INPUT_FLOAT(point_extras);
    TORCH_CHECK(
        (keypoints.dim() == 2 && keypoints.size(1) == 3) ||
        (keypoints.dim() == 3 && keypoints.size(2) == 3),
        "keypoints must be (M,3) or (B,M,3)"
    );
    TORCH_CHECK(
        (points.dim() == 2 && points.size(1) == 3) ||
        (points.dim() == 3 && points.size(2) == 3),
        "points must be (N,3) or (B,N,3)"
    );
    TORCH_CHECK(keypoints.dim() == points.dim(),
        "keypoints and points must both be batched or both be unbatched");

    const bool batched = (keypoints.dim() == 3);
    const int64_t B = batched ? keypoints.size(0) : 1;
    const int64_t M_per_batch = batched ? keypoints.size(1) : keypoints.size(0);
    const int64_t N_per_batch = batched ? points.size(1) : points.size(0);
    if (batched) {
        TORCH_CHECK(keypoints.size(0) == points.size(0), "keypoints/points batch mismatch");
    }
    auto keypoints_flat = batched ? keypoints.view({-1, 3}) : keypoints;
    auto points_flat = batched ? points.view({-1, 3}) : points;

    auto check_f = [](const torch::Tensor& t, const char* name){
        if (t.defined() && t.numel() > 0) {
            TORCH_CHECK(t.device().is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
            TORCH_CHECK(t.dtype() == torch::kFloat, name, " must be float32");
        }
    };
    auto flatten_last = [](const torch::Tensor& t) -> torch::Tensor {
        if (!t.defined()) return t;
        if (t.dim() == 3) return t.view({-1, t.size(2)});
        return t;
    };
    auto check_layout = [&](const torch::Tensor& t, const char* name, bool for_keypoints, bool allow_empty_last_dim){
        if (!(t.defined() && t.numel() > 0)) return;
        const int64_t expected_rows = for_keypoints ? M_per_batch : N_per_batch;
        if (batched) {
            TORCH_CHECK(t.dim() == 3, name, " must be 3D for batched inputs");
            TORCH_CHECK(t.size(0) == B && t.size(1) == expected_rows, name, " shape mismatch");
        } else {
            TORCH_CHECK(t.dim() == 2, name, " must be 2D for unbatched inputs");
            TORCH_CHECK(t.size(0) == expected_rows, name, " shape mismatch");
        }
        if (!allow_empty_last_dim) {
            TORCH_CHECK(t.size(-1) > 0, name, " has invalid last dimension");
        }
    };

    check_f(key_feat, "key_feat");
    check_f(point_kfeat, "point_kfeat");
    check_f(point_vfeat, "point_vfeat");
    if (key_feat.defined() && key_feat.numel() > 0 && point_kfeat.defined() && point_kfeat.numel() > 0) {
        TORCH_CHECK(key_feat.dtype() == point_kfeat.dtype(), "key_feat and point_kfeat must be the same dtype");
    }
    if (key_feat.defined() && key_feat.numel() > 0 && point_vfeat.defined() && point_vfeat.numel() > 0) {
        TORCH_CHECK(key_feat.dtype() == point_vfeat.dtype(), "key_feat and point_vfeat must be the same dtype");
    }
    if (point_kfeat.defined() && point_kfeat.numel() > 0 && point_vfeat.defined() && point_vfeat.numel() > 0) {
        TORCH_CHECK(point_kfeat.dtype() == point_vfeat.dtype(), "point_kfeat and point_vfeat must be the same dtype");
    }

    if (key_feat.defined() && key_feat.numel() > 0) {
        check_layout(key_feat, "key_feat", true, false);
        if (point_kfeat.defined() && point_kfeat.numel() > 0) {
            check_layout(point_kfeat, "point_kfeat", false, false);
            TORCH_CHECK(point_kfeat.size(-1) == key_feat.size(-1), "point_kfeat and key_feat size mismatch");
        }
        if (point_vfeat.defined() && point_vfeat.numel() > 0) {
            check_layout(point_vfeat, "point_vfeat", false, false);
            TORCH_CHECK(point_vfeat.size(-1) == key_feat.size(-1), "point_vfeat and key_feat size mismatch");
        }
    }

    if ((point_vfeat.defined() && point_vfeat.numel() > 0) || (point_kfeat.defined() && point_kfeat.numel() > 0)) {
        check_layout(point_kfeat, "point_kfeat", false, false);
        check_layout(point_vfeat, "point_vfeat", false, false);
        if (point_kfeat.defined() && point_kfeat.numel() > 0 && point_vfeat.defined() && point_vfeat.numel() > 0) {
            TORCH_CHECK(point_kfeat.size(-1) == point_vfeat.size(-1), "point_kfeat and point_vfeat size mismatch");
        }
    }
    if (point_extras.defined() && point_extras.numel() > 0) {
        check_layout(point_extras, "point_extras", false, true);
    }

    int64_t C = 0;
    if (key_feat.defined() && key_feat.numel() > 0) C = key_feat.size(-1);
    else if (point_kfeat.defined() && point_kfeat.numel() > 0) C = point_kfeat.size(-1);
    else if (point_vfeat.defined() && point_vfeat.numel() > 0) C = point_vfeat.size(-1);
    else if (weights.defined() && weights.numel() > 0) {
        const int share_factor = (proj_share_group == 4) ? 4 : 1;
        C = weights.size(0) * weights.size(3) * share_factor;
    }
    if (C > 0) TORCH_CHECK(C % head_dim == 0, "C must be divisible by head_dim");
    const int64_t H = (C > 0) ? (C / head_dim) : 0;
    if (proj_share_group == 4 && H > 0) TORCH_CHECK((H % 4) == 0, "C/head_dim must be divisible by 4 when proj_share_group=4");

    if (weights.defined() && weights.numel() > 0) {
        check_f(weights, "weights");
        const int64_t expected_weight_h = (proj_share_group == 4) ? (H / 4) : H;
        TORCH_CHECK(weights.size(0) == expected_weight_h, "weights H mismatch");
        TORCH_CHECK(weights.size(1) == 3, "weights must have 3 biases (q, k, v)");
        TORCH_CHECK(weights.size(3) == head_dim, "weights HEAD_DIM mismatch");
        if (point_extras.defined() && point_extras.numel() > 0) {
            TORCH_CHECK(point_extras.size(-1) <= 4, "point_extras P must be <= 4");
        }
    }
    if (bias.defined() && bias.numel() > 0) {
        check_f(bias, "bias");
        TORCH_CHECK(weights.defined() && weights.numel() > 0, "bias requires weights to be provided");
        TORCH_CHECK(bias.dim() == 3 && bias.size(1) == 3 && bias.size(2) == head_dim, "bias must be (H,3,d_head)");
        TORCH_CHECK(bias.size(0) == H, "bias H mismatch");
    }

    auto key_feat_flat = flatten_last(key_feat);
    auto point_kfeat_flat = flatten_last(point_kfeat);
    auto point_vfeat_flat = flatten_last(point_vfeat);
    auto point_extras_flat = flatten_last(point_extras);

    torch::Tensor nn_idx;
    torch::Tensor nn_count;
    torch::Tensor nn_idx_flat;
    torch::Tensor nn_count_flat;
    int K_use = K;

    if (has_nn_idx) {
        CHECK_INPUT_INT(nn_idx_in);
        CHECK_INPUT_INT(nn_count_in);
        if (batched) {
            TORCH_CHECK(nn_idx_in.dim() == 3 && nn_idx_in.size(0) == B && nn_idx_in.size(1) == M_per_batch,
                "nn_idx must be (B,M,K) for batched inputs");
            TORCH_CHECK(nn_count_in.dim() == 2 && nn_count_in.size(0) == B && nn_count_in.size(1) == M_per_batch,
                "nn_count must be (B,M) for batched inputs");
            K_use = nn_idx_in.size(2);
            nn_idx = nn_idx_in.contiguous();
            nn_count = nn_count_in.contiguous();
            nn_idx_flat = nn_idx.view({-1, K_use});
            nn_count_flat = nn_count.view({-1});
        } else {
            TORCH_CHECK(nn_idx_in.dim() == 2 && nn_idx_in.size(0) == M_per_batch,
                "nn_idx must be (M,K) for unbatched inputs");
            TORCH_CHECK(nn_count_in.dim() == 1 && nn_count_in.size(0) == M_per_batch,
                "nn_count must be (M,) for unbatched inputs");
            K_use = nn_idx_in.size(1);
            nn_idx = nn_idx_in.contiguous();
            nn_count = nn_count_in.contiguous();
            nn_idx_flat = nn_idx;
            nn_count_flat = nn_count;
        }
        TORCH_CHECK(K_use >= 1, "nn_idx must have K >= 1");
        if (nn_count_flat.numel() > 0) {
            const int max_count = nn_count_flat.max().item<int>();
            const int min_count = nn_count_flat.min().item<int>();
            TORCH_CHECK(min_count >= 0, "nn_count must be >= 0");
            TORCH_CHECK(max_count <= K_use, "nn_count values must be <= nn_idx.shape[-1]");
        }
        nn_idx_flat = std::get<0>(nn_idx_flat.sort(/*dim=*/-1, /*descending=*/true));
        if (batched) nn_idx = nn_idx_flat.view({B, M_per_batch, K_use});
        else nn_idx = nn_idx_flat;
    } else {
        std::vector<torch::Tensor> knn = cuda_pcd_knn(keypoints, points, K_use, radius);
        nn_idx = knn[0];
        nn_count = knn[1];
        nn_idx_flat = batched ? nn_idx.view({-1, K_use}) : nn_idx;
        nn_idx_flat = std::get<0>(nn_idx_flat.sort(/*dim=*/-1, /*descending=*/true));
        nn_count_flat = batched ? nn_count.view({-1}) : nn_count;
        if (batched) nn_idx = nn_idx_flat.view({B, M_per_batch, K_use});
        else nn_idx = nn_idx_flat;
    }

    attn_out out_flat = cuda_pcd_attn_fwd(
        keypoints_flat, key_feat_flat,
        points_flat, point_kfeat_flat, point_vfeat_flat, point_extras_flat,
        nn_idx_flat, nn_count_flat, weights, bias,
        head_dim, K_use, coord_scale, rel_clamp, proj_share_group
    );

    auto out = out_flat.out;
    auto head_max = out_flat.head_max;
    auto head_denom = out_flat.head_denom;
    if (batched) {
        out = out.view({B, M_per_batch, C});
        head_max = head_max.view({B, M_per_batch, H});
        head_denom = head_denom.view({B, M_per_batch, H});
    }

    return {out, nn_idx, nn_count, head_max, head_denom};
}

std::vector<torch::Tensor> pcd_attn_bwd(
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
#ifdef PCD_ATTN_ONLY_HD96
    TORCH_CHECK(head_dim == 96, "This build only supports head_dim=96");
#else
    TORCH_CHECK(
        head_dim == 48 || head_dim == 72 || head_dim == 96,
        "Supported head_dim values are {48,72,96}"
    );
#endif
    TORCH_CHECK(coord_scale >= 0.0, "coord_scale must be non-negative");
    TORCH_CHECK(rel_clamp >= 0.0, "rel_clamp must be non-negative");
    TORCH_CHECK(split_q >= 1, "split_q must be >= 1");
    TORCH_CHECK(split_k >= 1, "split_k must be >= 1");
    TORCH_CHECK(proj_share_group == 1 || proj_share_group == 4, "proj_share_group must be 1 or 4");

    CHECK_INPUT_FLOAT(keypoints);
    CHECK_INPUT_FLOAT(points);
    CHECK_INPUT_FLOAT(point_extras);
    TORCH_CHECK(
        (keypoints.dim() == 2 && keypoints.size(1) == 3) ||
        (keypoints.dim() == 3 && keypoints.size(2) == 3),
        "keypoints must be (M,3) or (B,M,3)"
    );
    TORCH_CHECK(
        (points.dim() == 2 && points.size(1) == 3) ||
        (points.dim() == 3 && points.size(2) == 3),
        "points must be (N,3) or (B,N,3)"
    );
    TORCH_CHECK(keypoints.dim() == points.dim(),
        "keypoints and points must both be batched or both be unbatched");

    const bool batched = (keypoints.dim() == 3);
    const int64_t B = batched ? keypoints.size(0) : 1;
    const int64_t M_per_batch = batched ? keypoints.size(1) : keypoints.size(0);
    const int64_t N_per_batch = batched ? points.size(1) : points.size(0);
    if (batched) TORCH_CHECK(keypoints.size(0) == points.size(0), "keypoints/points batch mismatch");

    TORCH_CHECK(
        (batched && nn_idx.dim() == 3 && nn_idx.size(0) == B && nn_idx.size(1) == M_per_batch) ||
        (!batched && nn_idx.dim() == 2 && nn_idx.size(0) == M_per_batch),
        "nn_idx must be (M,K) or (B,M,K)"
    );
    TORCH_CHECK(
        (batched && nn_count.dim() == 2 && nn_count.size(0) == B && nn_count.size(1) == M_per_batch) ||
        (!batched && nn_count.dim() == 1 && nn_count.size(0) == M_per_batch),
        "nn_count must be (M,) or (B,M)"
    );
    CHECK_INPUT_INT(nn_idx); CHECK_INPUT_INT(nn_count);
    CHECK_INPUT_FLOAT(head_max); CHECK_INPUT_FLOAT(head_denom);

    auto check_f = [](const torch::Tensor& t, const char* name){
        if (t.defined() && t.numel() > 0) {
            TORCH_CHECK(t.device().is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
            TORCH_CHECK(t.dtype() == torch::kFloat, name, " must be float32");
        }
    };
    auto flatten_last = [](const torch::Tensor& t) -> torch::Tensor {
        if (!t.defined()) return t;
        if (t.dim() == 3) return t.view({-1, t.size(2)});
        return t;
    };
    auto flatten_count = [](const torch::Tensor& t) -> torch::Tensor {
        if (!t.defined()) return t;
        if (t.dim() == 2) return t.view({-1});
        return t;
    };

    check_f(grad_out, "grad_out");
    check_f(out, "out");
    check_f(key_feat, "key_feat");
    check_f(point_kfeat, "point_kfeat");
    check_f(point_vfeat, "point_vfeat");
    if (key_feat.defined() && key_feat.numel() > 0 && point_kfeat.defined() && point_kfeat.numel() > 0) {
        TORCH_CHECK(key_feat.dtype() == point_kfeat.dtype(), "key_feat and point_kfeat must be the same dtype");
    }
    if (key_feat.defined() && key_feat.numel() > 0 && point_vfeat.defined() && point_vfeat.numel() > 0) {
        TORCH_CHECK(key_feat.dtype() == point_vfeat.dtype(), "key_feat and point_vfeat must be the same dtype");
    }
    if (point_kfeat.defined() && point_kfeat.numel() > 0 && point_vfeat.defined() && point_vfeat.numel() > 0) {
        TORCH_CHECK(point_kfeat.dtype() == point_vfeat.dtype(), "point_kfeat and point_vfeat must be the same dtype");
    }

    auto keypoints_flat = batched ? keypoints.view({-1, 3}) : keypoints;
    auto points_flat = batched ? points.view({-1, 3}) : points;
    const int64_t M_total = keypoints_flat.size(0);
    const int64_t N_total = points_flat.size(0);

    auto grad_out_flat = flatten_last(grad_out);
    auto out_flat = flatten_last(out);
    auto key_feat_flat = flatten_last(key_feat);
    auto point_kfeat_flat = flatten_last(point_kfeat);
    auto point_vfeat_flat = flatten_last(point_vfeat);
    auto point_extras_flat = flatten_last(point_extras);
    auto nn_idx_flat = batched ? nn_idx.view({-1, nn_idx.size(2)}) : nn_idx;
    auto nn_count_flat = flatten_count(nn_count);
    auto head_max_flat = flatten_last(head_max);
    auto head_denom_flat = flatten_last(head_denom);

    TORCH_CHECK(out_flat.dtype() == torch::kFloat, "out must be float32");
    TORCH_CHECK(
        grad_out_flat.dim() == 2 && grad_out_flat.size(0) == M_total && grad_out_flat.size(1) % head_dim == 0,
        "grad_out must be (M, C) or (B, M, C)"
    );
    int C = grad_out_flat.size(1);
    int H = C / head_dim;
    if (proj_share_group == 4) TORCH_CHECK((H % 4) == 0, "C/head_dim must be divisible by 4 when proj_share_group=4");
    TORCH_CHECK(out_flat.dim() == 2 && out_flat.size(0) == M_total && out_flat.size(1) == C, "out shape mismatch");

    if (key_feat_flat.defined() && key_feat_flat.numel() > 0) {
        TORCH_CHECK(key_feat_flat.dim() == 2 && key_feat_flat.size(0) == M_total && key_feat_flat.size(1) == C, "key_feat shape mismatch");
        TORCH_CHECK(key_feat_flat.dtype() == grad_out_flat.dtype(), "key_feat and grad_out must be the same dtype");
    }
    if (point_kfeat_flat.defined() && point_kfeat_flat.numel() > 0) {
        TORCH_CHECK(point_kfeat_flat.dim() == 2 && point_kfeat_flat.size(0) == N_total && point_kfeat_flat.size(1) == C, "point_kfeat shape mismatch");
    }
    if (point_vfeat_flat.defined() && point_vfeat_flat.numel() > 0) {
        TORCH_CHECK(point_vfeat_flat.dim() == 2 && point_vfeat_flat.size(0) == N_total && point_vfeat_flat.size(1) == C, "point_vfeat shape mismatch");
    }
    if (point_extras_flat.defined() && point_extras_flat.numel() > 0) {
        TORCH_CHECK(
            point_extras_flat.dim() == 2 && point_extras_flat.size(0) == N_total && point_extras_flat.size(1) <= 4,
            "point_extras must be (N,P) or (B,N,P)"
        );
    }

    if (weights.defined() && weights.numel() > 0) {
        check_f(weights, "weights");
        const int expected_weight_h = (proj_share_group == 4) ? (H / 4) : H;
        TORCH_CHECK(weights.size(0) == expected_weight_h, "weights H mismatch");
        TORCH_CHECK(weights.size(1) == 3, "weights must have 3 biases (q, k, v)");
        TORCH_CHECK(weights.size(3) == head_dim, "weights HEAD_DIM mismatch");
    }

    if (bias.defined() && bias.numel() > 0) {
        check_f(bias, "bias");
        TORCH_CHECK(weights.defined() && weights.numel() > 0, "bias requires weights to be provided");
        TORCH_CHECK(bias.dtype() == weights.dtype(), "bias and weights must be the same dtype");
        TORCH_CHECK(bias.dim() == 3 && bias.size(1) == 3 && bias.size(2) == head_dim, "bias must be (H,3,d_head)");
        TORCH_CHECK(bias.size(0) == H, "bias H mismatch");
    }

    std::vector<torch::Tensor> grads = cuda_pcd_attn_bwd(
        grad_out_flat, key_feat_flat, keypoints_flat, points_flat, point_kfeat_flat, point_vfeat_flat, point_extras_flat,
        nn_idx_flat, nn_count_flat, head_max_flat, head_denom_flat, weights, bias, out_flat,
        head_dim, coord_scale, rel_clamp, split_q, split_k, proj_share_group
    );

    if (batched) {
        if (grads[0].defined() && grads[0].dim() == 2 && key_feat.dim() == 3) {
            grads[0] = grads[0].view({B, M_per_batch, key_feat.size(2)});
        }
        if (grads[1].defined() && grads[1].dim() == 2 && point_kfeat.dim() == 3) {
            grads[1] = grads[1].view({B, N_per_batch, point_kfeat.size(2)});
        }
        if (grads[2].defined() && grads[2].dim() == 2 && point_vfeat.dim() == 3) {
            grads[2] = grads[2].view({B, N_per_batch, point_vfeat.size(2)});
        }
    }

    return grads;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_indices", &conv_indices, "make indices for conv");
  m.def("conv_fwd", &conv_fwd, "forward grouped conv");
  m.def("conv_dw", &conv_dw, "backward weights");
  m.def("conv_dx", &conv_dx, "backward input");
  m.def("dense_conv_fwd", &dense_conv_fwd, "forward dense conv");
  m.def("dense_conv_dx", &dense_conv_dx, "backward input dense conv");
  m.def("dense_conv_dw", &dense_conv_dw, "backward weights dense conv");
  m.def("vox_connected_components", &vox_connected_components, "6-neighbor connected components");
  m.def("vox_hierarchical_patches", &vox_hierarchical_patches, "hierarchical locality DSU patching");
  m.def("pcd_knn", &pcd_knn, "point cloud K nearest neighbours");
  m.def("pcd_vox_knn", &pcd_vox_knn, "sparse voxel neighbourhood indices for keypoints");
  m.def(
      "pcd_attn",
      &pcd_attn,
      "point cloud cross attention forward",
      py::arg("keypoints"),
      py::arg("key_feat"),
      py::arg("points"),
      py::arg("point_kfeat"),
      py::arg("point_vfeat"),
      py::arg("point_extras"),
      py::arg("weights"),
      py::arg("bias"),
      py::arg("head_dim"),
      py::arg("K"),
      py::arg("radius"),
      py::arg("coord_scale"),
      py::arg("rel_clamp") = 0.0f,
      py::arg("proj_share_group"),
      py::arg("nn_idx") = torch::Tensor(),
      py::arg("nn_count") = torch::Tensor());
  m.def("pcd_attn_bwd", &pcd_attn_bwd, "point cloud cross attention backward");
}
