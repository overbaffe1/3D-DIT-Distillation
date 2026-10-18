

#include <mesh.h>
#include <cutil_math.h>


__global__ void render_sign_kernel(
    const long n,
    const long n_views,
    const long resy,
    const long resx,
    const float *points,
    const float *depth,
    const float *w2cs,
    const float *fovs,
    float* out_sign
){

    const long thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= n)
        return;

    const long gidx = thread_idx;

    float sign = -1.0f;
    float3 pos = make_float3(points[3 * gidx + 0], points[3 * gidx + 1], points[3 * gidx + 2]);

    for(size_t view_idx = 0; view_idx < n_views; view_idx++){

        float3 campos = make_float3(w2cs[16 * view_idx + 0 * 4 + 3], w2cs[16 * view_idx + 1 * 4 + 3], w2cs[16 * view_idx + 2 * 4 + 3]);
        campos += make_float3(w2cs[16 * view_idx + 0 * 4 + 0], w2cs[16 * view_idx + 1 * 4 + 0], w2cs[16 * view_idx + 2 * 4 + 0]) * pos.x;
        campos += make_float3(w2cs[16 * view_idx + 0 * 4 + 1], w2cs[16 * view_idx + 1 * 4 + 1], w2cs[16 * view_idx + 2 * 4 + 1]) * pos.y;
        campos += make_float3(w2cs[16 * view_idx + 0 * 4 + 2], w2cs[16 * view_idx + 1 * 4 + 2], w2cs[16 * view_idx + 2 * 4 + 2]) * pos.z;

        float curr_depth = -campos.z;
        float side = tanf(fovs[view_idx] / 2.0f);
        int x = floorf(((campos.x / curr_depth / side) / 2.0f + 0.5f) * resx);
        int y = floorf(((-campos.y / curr_depth / side) / 2.0f + 0.5f) * resy);

        if((x >= 0) && (y >= 0) && (x <= resx - 1) && (y <= resy - 1)){
            float gt_depth = depth[
                view_idx * resx * resy
                + y * resx
                + x
            ];
            if (gt_depth > curr_depth) {
                sign = 1.0f;
                break;
            }
        }
    }

    out_sign[gidx] = sign;
}

torch::Tensor cuda_render_sign(torch::Tensor points, torch::Tensor depth, torch::Tensor w2cs, torch::Tensor fovs) {
    const long n = points.size(0);
    const long n_views = depth.size(0), resy = depth.size(1), resx = depth.size(2);
    const int threads = 512;

    torch::Tensor out_sign = torch::empty(
        {(long int) n},
        torch::TensorOptions().dtype(depth.dtype()).device(depth.device())
    );

    render_sign_kernel<<<(n + threads - 1) / threads, threads>>>(
        n,
        n_views,
        resy,
        resx,
        points.data_ptr<float>(),
        depth.data_ptr<float>(),
        w2cs.data_ptr<float>(),
        fovs.data_ptr<float>(),
        out_sign.data_ptr<float>()
    );

    return out_sign;
}

__global__ void project_features_kernel(
    const int b,
    const int n,
    const int n_views,
    const int resy,
    const int resx,
    const int n_features,
    const float *coords,
    const float *features,
    const float *w2cs,
    const float *fovs,
    float* out_features
){

    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= b * n * n_features)
        return;

    const int bidx = (thread_idx / (n * n_features)) % b;
    const int coord_idx = (thread_idx / n_features) % n;
    const int feat_idx = thread_idx % n_features;

    float value = 0.0f, denom = 0.0f;
    float3 pos = make_float3(coords[bidx * n * 3 + 3 * coord_idx + 0], coords[bidx * n * 3 + 3 * coord_idx + 1], coords[bidx * n * 3 + 3 * coord_idx + 2]);

    const float* w2c = w2cs + bidx * n_views * 4 * 4;
    const float* feature = features + bidx * resy * resx * n_views * n_features;
    for(size_t view_idx = 0; view_idx < n_views; view_idx++){

        float3 campos = make_float3(w2c[16 * view_idx + 0 * 4 + 3], w2c[16 * view_idx + 1 * 4 + 3], w2c[16 * view_idx + 2 * 4 + 3]);
        campos += make_float3(w2c[16 * view_idx + 0 * 4 + 0], w2c[16 * view_idx + 1 * 4 + 0], w2c[16 * view_idx + 2 * 4 + 0]) * pos.x;
        campos += make_float3(w2c[16 * view_idx + 0 * 4 + 1], w2c[16 * view_idx + 1 * 4 + 1], w2c[16 * view_idx + 2 * 4 + 1]) * pos.y;
        campos += make_float3(w2c[16 * view_idx + 0 * 4 + 2], w2c[16 * view_idx + 1 * 4 + 2], w2c[16 * view_idx + 2 * 4 + 2]) * pos.z;

        float depth = -campos.z;
        float side = tanf(fovs[bidx * n_views + view_idx] / 2.0f);
        int x = floorf(((campos.x / depth / side) / 2.0f + 0.5f) * resx);
        int y = floorf(((-campos.y / depth / side) / 2.0f + 0.5f) * resy);

        if((x >= 0) && (y >= 0) && (x <= resx - 1) && (y <= resy - 1)){
            value += feature[
                  y * resx * n_views * n_features
                + x * n_views * n_features
                + view_idx * n_features
                + feat_idx
            ];
            denom += 1.0f;
        }
    }

    out_features[bidx * n * n_features + coord_idx * n_features + feat_idx] = value / fmaxf(denom, 1.0f);
}

torch::Tensor cuda_project_features(torch::Tensor coords, torch::Tensor features, torch::Tensor w2cs, torch::Tensor fovs) {
    const int b = coords.size(0), n = coords.size(1);
    const int resy = features.size(1), resx = features.size(2), n_views = features.size(3), n_features = features.size(4);
    const int threads = 512;

    torch::Tensor out_features = torch::empty(
        {b, n, n_features},
        torch::TensorOptions().dtype(features.dtype()).device(features.device())
    );

    project_features_kernel<<<(b * n * n_features + threads - 1) / threads, threads>>>(
        b,
        n,
        n_views,
        resy,
        resx,
        n_features,
        coords.data_ptr<float>(),
        features.data_ptr<float>(),
        w2cs.data_ptr<float>(),
        fovs.data_ptr<float>(),
        out_features.data_ptr<float>()
    );

    return out_features;
}