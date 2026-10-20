/*
  Copyright (C) 2022-present Naver Corporation. All rights reserved.
  Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
*/

#include <torch/extension.h>

// forward declaration
void rope_nd_cuda( torch::Tensor tokens, const torch::Tensor pos, const float base, const float fwd, torch::Tensor out);

void rope_nd_cpu( torch::Tensor tokens, const torch::Tensor positions, const float base, const float fwd, torch::Tensor out)
{
    const int B = tokens.size(0);
    const int N = tokens.size(1);
    const int H = tokens.size(2);
    const int DIM = positions.size(2);
    const int D = tokens.size(3) / (2 * DIM);

    auto tok = tokens.accessor<float, 4>();
    auto res = out.accessor<float, 4>();
    auto pos = positions.accessor<float, 3>();

    for (int b = 0; b < B; b++) {
      for (int x = 0; x < DIM; x++) { // y and then x (2d)
        for (int n = 0; n < N; n++) {

            // grab the token position
            const float p = pos[b][n][x];

            for (int h = 0; h < H; h++) {
                for (int d = 0; d < D; d++) {
                    // grab the two values
                    float u = tok[b][n][h][d+0+x*2*D];
                    float v = tok[b][n][h][d+D+x*2*D];

                    // grab the cos,sin
                    const float inv_freq = fwd * p / powf(base, d/float(D));
                    float c = cosf(inv_freq);
                    float s = sinf(inv_freq);

                    // write the result
                    res[b][n][h][d+0+x*2*D] = u*c - v*s;
                    res[b][n][h][d+D+x*2*D] = v*c + u*s;
                }
            }
        }
      }
    }
}

void rope_nd( torch::Tensor tokens,     // B,N,H,D
        const torch::Tensor positions,  // B,N,DIM
        const float base,
        const float fwd,
        torch::Tensor out)
{
    TORCH_CHECK(tokens.dim() == 4, "tokens must have 4 dimensions");
    TORCH_CHECK(positions.dim() == 3, "positions must have 3 dimensions");
    TORCH_CHECK(tokens.size(0) == positions.size(0), "batch size differs between tokens & positions");
    TORCH_CHECK(tokens.size(1) == positions.size(1), "seq_length differs between tokens & positions");
    TORCH_CHECK(tokens.is_cuda() == positions.is_cuda(), "tokens and positions are not on the same device" );
    TORCH_CHECK(positions.dtype() == torch::kFloat, "positions must be float");

    if (tokens.is_cuda())
        rope_nd_cuda( tokens, positions, base, fwd, out);
    else
        rope_nd_cpu( tokens, positions, base, fwd, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_nd", &rope_nd, "RoPE nd forward/backward");
}