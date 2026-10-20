/*
  Copyright (C) 2022-present Naver Corporation. All rights reserved.
  Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(tensor) {\
    TORCH_CHECK((tensor).is_cuda(), #tensor " is not in cuda memory"); \
    TORCH_CHECK((tensor).is_contiguous(), #tensor " is not contiguous"); }
void CHECK_KERNEL() {auto error = cudaGetLastError(); TORCH_CHECK( error == cudaSuccess, cudaGetErrorString(error));}


template < typename scalar_t  >
__global__ void rope_nd_cuda_kernel(
        const int N, const int H, const int D,
        const scalar_t* __restrict__ tokens,
//         torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> tokens,
        const float* __restrict__ pos,
        const int dim,
        const float base,
        const float fwd,
        scalar_t* __restrict__ out
//         torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> out
)
        // const int N, const int H, const int D )
{
    // tokens shape = (B, N, H, D)
//     const int N = tokens.size(1);
//     const int H = tokens.size(2);
//     const int D = tokens.size(3);

    // each block update a single token, for all heads
    // each thread takes care of a single output
    extern __shared__ double shared[];
    double* shared_inv_freq = shared + D;

    const int b = blockIdx.x / N;
    const int n = blockIdx.x % N;

    const int Q = D / (2 * dim);
    // one token = [0..Q : Q..2Q : 2Q..3Q : 3Q..D]
    //              u_Y     v_Y     u_X      v_X

    // shared memory: first, compute inv_freq
    if (threadIdx.x < Q){
        shared_inv_freq[threadIdx.x] = static_cast<double>(fwd * pow(static_cast<double>(base), -static_cast<double>(threadIdx.x) / Q));
    }
    __syncthreads();

    // start of X or Y part
    const int X = threadIdx.x / (D / dim);
    const int m = (X*D/dim) + (threadIdx.x % Q);   // index of u_Y or u_X

    // grab the cos,sin appropriate for me
    const double freq = pos[blockIdx.x*dim+X] * shared_inv_freq[threadIdx.x % Q];
    const double freq_cos = cos(freq);
    const double freq_sin = sin(freq);
    /*
    float* shared_cos_sin = shared + D + D/4;
    if ((threadIdx.x % (D/2)) < Q)
        shared_cos_sin[m+0] = cosf(freq);
    else
        shared_cos_sin[m+Q] = sinf(freq);
    __syncthreads();
    const float cos = shared_cos_sin[m+0];
    const float sin = shared_cos_sin[m+Q];
    */

    for (int h = 0; h < H; h++)
    {
        // then, load all the token for this head in shared memory
        shared[threadIdx.x] = static_cast<double>(tokens[static_cast<size_t>(b) * N * H * D + static_cast<size_t>(n) * H * D + h * D + threadIdx.x]);
        __syncthreads();

        const double u = shared[m];
        const double v = shared[m+Q];

        // write output
        scalar_t* out_item = out + (static_cast<size_t>(b) * N * H * D + static_cast<size_t>(n) * H * D + h * D + threadIdx.x);
        if ((threadIdx.x % (D/dim)) < Q)
            *out_item = static_cast<scalar_t>(u*freq_cos - v*freq_sin);
        else
            *out_item = static_cast<scalar_t>(v*freq_cos + u*freq_sin);
    }
}

void rope_nd_cuda( torch::Tensor tokens, const torch::Tensor pos, const float base, const float fwd, torch::Tensor out)
{
    const int B = tokens.size(0); // batch size
    const int N = tokens.size(1); // sequence length
    const int H = tokens.size(2); // number of heads
    const int D = tokens.size(3); // dimension per head
    const int DIM = pos.size(2); // number of dimensions

    TORCH_CHECK(tokens.is_contiguous(), "tokens are not contiguous");
    TORCH_CHECK(pos.is_contiguous(), "positions are not contiguous");
    TORCH_CHECK(pos.size(0) == B && pos.size(1) == N && pos.size(2) == DIM, "bad pos.shape");
    TORCH_CHECK(D % (2 * DIM) == 0, "token dim must be multiple of 4");

    // one block for each layer, one thread per local-max
    const int THREADS_PER_BLOCK = D;
    const int N_BLOCKS = B * N; // each block takes care of H*D values
    const int SHARED_MEM = sizeof(double) * (D + D/(2 * DIM));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, tokens.scalar_type(), "rope_nd_cuda", ([&] {
        rope_nd_cuda_kernel<scalar_t> <<<N_BLOCKS, THREADS_PER_BLOCK, SHARED_MEM>>> (
            N, H, D,
            tokens.data_ptr<scalar_t>(),
//             tokens.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            pos.data_ptr<float>(), DIM,
            base, fwd,
            out.data_ptr<scalar_t>()
//             out.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
            ); //, N, H, D );
    }));
}