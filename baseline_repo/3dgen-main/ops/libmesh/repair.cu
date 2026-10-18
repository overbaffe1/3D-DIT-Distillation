#include <cutil_math.h>
#include <mesh.h>

__global__ void build_edge_code_kernel(
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

        const int64_t i0 = static_cast<int64_t>(min(fids[k], fids[(k + 1) % 3]));
        const int64_t i1 = static_cast<int64_t>(max(fids[k], fids[(k + 1) % 3]));

        codes[k * N_FACES + gidx] = (i0 << 31) | i1;
    }

}

__global__ void gather_verts_ids_kernel(
    const int N_EDGES,
    const int64_t* __restrict__ edges,
    int2* __restrict__ ids
 ){

    const int gidx = global_idx();
    if(gidx >= N_EDGES) return;

    const int64_t edge = edges[gidx];
    const int i0 = static_cast<int32_t>(edge >> 31), i1 = static_cast<int32_t>(edge & ((1LL << 31) - 1));
    ids[gidx] = make_int2(i0, i1);

}

__global__ void build_half_edge_kernel(
    const int N_EDGES,
    const int2* __restrict__ edges,
    int64_t* __restrict__ half_edges,
    int* __restrict__ rev
) {

    const int gidx = global_idx();
    if(gidx >= N_EDGES) return;

    const int2 edge = edges[gidx];
    const int64_t i0 = static_cast<int64_t>(edge.x), i1 = static_cast<int64_t>(edge.y);

    half_edges[0 * N_EDGES + gidx] = (i0 << 31) | i1;
    half_edges[1 * N_EDGES + gidx] = (i1 << 31) | i0;

    rev[0 * N_EDGES + gidx] = 1 * N_EDGES + gidx;
    rev[1 * N_EDGES + gidx] = 0 * N_EDGES + gidx;
}

__global__ void build_verts_offset_kernel(
    const int N_HALF_EDGES,
    const int64_t* __restrict__ half_edges,
    int* __restrict__ voff
) {

    const int gidx = global_idx();
    if (gidx >= N_HALF_EDGES) return;

    const int64_t edge = half_edges[gidx];
    const int i0 = static_cast<int32_t>(edge >> 31);
    atomicAdd(&voff[i0], static_cast<int32_t>(1));

}

__global__ void build_cycle_code_kernel(
    const int N_HALF_EDGES,
    const int64_t* __restrict__ half_edges,
    const int* __restrict__ cycle_idx,
    int64_t* __restrict__ cycle_code
) {

    const int gidx = global_idx();
    if(gidx >= N_HALF_EDGES) return;

    const int64_t i0 = half_edges[gidx] >> 31;
    const int64_t idx = static_cast<int64_t>(cycle_idx[gidx]);

    cycle_code[gidx] = (i0 << 31) | idx;

}

__device__ __forceinline__ int32_t get_cycle(const int64_t cycle_code){
    return static_cast<int32_t>(cycle_code & ((1LL << 31) - 1));
}


__global__ void fix_large_cycle_kernel(
    const int N_BOUNDARY_VERTS,
    const int32_t* __restrict__ voff,
    const int32_t* __restrict__ cycle_idx,
    const int64_t* __restrict__ cycle_code,
    const int32_t* __restrict__ inv_map,
    const int32_t* __restrict__ reverse_edges,
    int32_t* __restrict__ pair_edges,
    int32_t* __restrict__ changed
) {
    const int gidx = global_idx();
    if (gidx >= N_BOUNDARY_VERTS) return;

    int vstart = 0;
    if(gidx > 0) vstart = voff[gidx - 1];
    int vend = voff[gidx];

    bool found_large_cycle = false;
    int prev_cycle = get_cycle(cycle_code[vstart]);
    for(int curr_idx = vstart + 1; curr_idx < vend; curr_idx++){

        int curr_cycle = get_cycle(cycle_code[curr_idx]);
        if(curr_cycle != prev_cycle) {
            prev_cycle = curr_cycle;
            continue;
        }

        if (cycle_idx[reverse_edges[inv_map[curr_idx]]] < curr_cycle) continue; // permute only main cycle

        found_large_cycle = true;

        // do any permutation so we finish cycle earlier and keep involution property
        int idx0 = curr_idx - 1;
        int prev_edge = inv_map[idx0]; int pair0 = pair_edges[prev_edge];
        do {
            int curr_edge = inv_map[curr_idx]; int curr_pair = pair_edges[curr_edge];
            pair_edges[prev_edge] = curr_pair; pair_edges[curr_pair] = prev_edge;
            prev_edge = curr_edge;
        } while(++curr_idx < vend && get_cycle(cycle_code[curr_idx]) == curr_cycle);
        pair_edges[prev_edge] = pair0; pair_edges[pair0] = prev_edge;

        if (curr_idx < vend) prev_cycle = get_cycle(cycle_code[curr_idx]);
    }

    if (found_large_cycle) atomicExch(changed, 1);

}


__global__ void remap_half_edges_kernel(
    const int N_HALF_EDGES,
    int64_t* __restrict__ half_edges,
    const int32_t* __restrict__ inv_verts_map
) {

    const int gidx = global_idx();
    if (gidx >= N_HALF_EDGES) return;

    const int64_t edge = half_edges[gidx];
    const int64_t i0 = edge >> 31, i1 = edge & ((1LL << 31) - 1);
    half_edges[gidx] = (static_cast<int64_t>(inv_verts_map[i0]) << 31) | static_cast<int64_t>(inv_verts_map[i1]);

}
__global__ void build_cycle_edge_code_kernel(
    const int N_CYCLE_EDGES,
    const int64_t* __restrict__ cycle_edges,
    const int32_t* __restrict__ cycle_map,
    int64_t* __restrict__ cycle_edge_code
) {

    const int gidx = global_idx();
    if (gidx >= N_CYCLE_EDGES) return;

    const int64_t cycle_idx = static_cast<int64_t>(cycle_map[gidx]);
    const int64_t i0 = cycle_edges[gidx] >> 31;

    cycle_edge_code[gidx] = (cycle_idx << 31) | i0;

}

__global__ void find_cycle_next_kernel(
    const int N_CYCLE_EDGES,
    const int64_t* __restrict__ cycle_edges,
    const int32_t* __restrict__ cycle_map,
    const int32_t* __restrict__ cycle_offset,
    int32_t* __restrict__ cycle_next
) {

    const int gidx = global_idx();
    if (gidx >= N_CYCLE_EDGES) return;

    int cidx = cycle_map[gidx];
    int left_bound = 0, right_bound = cycle_offset[cidx];
    if(cidx > 0) left_bound = cycle_offset[cidx - 1];

    const int64_t edge = cycle_edges[gidx];
    const int64_t i1 = edge & ((1LL << 31) - 1);
    int l = left_bound, r = right_bound;
    while(r - l > 1){
        int m = (l + r) >> 1;
        if ((cycle_edges[m] >> 31) <= i1) l = m;
        else r = m;
    }


    cycle_next[gidx] = l;

}

__global__ void reorder_cycle_edges_kernel(
    const int N_CYCLES,
    const int32_t* __restrict__ cycle_offset,
    const int64_t* __restrict__ cycle_edges,
    const int32_t* __restrict__ cycle_next,
    int64_t* __restrict__ out
) {

    int cidx = global_idx();
    if (cidx >= N_CYCLES) return;

    int start = 0;
    if (cidx > 0) start = cycle_offset[cidx - 1];

    int pnt = 0;
    out[start + pnt++] = cycle_edges[start];

    int curr_idx = cycle_next[start];
    do {

        out[start + pnt++] = cycle_edges[curr_idx];
        curr_idx = cycle_next[curr_idx];

    } while (curr_idx != start);

}

__global__ void count_fill_triangles_kernel(
    const int N_CYCLES,
    const int32_t* __restrict__ cycle_off,
    int32_t* __restrict__ fill_tri_cnt,
    int32_t* __restrict__ fill_vrt_cnt
) {

    const int cidx = global_idx();
    if (cidx >= N_CYCLES) return;

    int start = 0, end = cycle_off[cidx];
    if (cidx > 0) start = cycle_off[cidx - 1];
    const int csize = end - start;

    if (csize == 3){
        fill_vrt_cnt[cidx] = 0;
        fill_tri_cnt[cidx] = 1;
    }
    else {
        fill_vrt_cnt[cidx] = 1;
        fill_tri_cnt[cidx] = csize;
    }

}


__global__ void make_fill_verts_kernel(
    const int N_CYCLE_EDGES,
    const int64_t* __restrict__ cycle_edges,
    const int32_t* __restrict__ cycle_map,
    const int32_t* __restrict__ cycle_off,
    const int32_t* __restrict__ fill_vrt_off,
    const float3* __restrict__ verts,
    float* __restrict__ cycle_center
){

    const int gidx = global_idx();
    if (gidx >= N_CYCLE_EDGES) return;

    const int cidx = cycle_map[gidx];

    int voff = 0;
    if (cidx > 0) voff = fill_vrt_off[cidx - 1];
    if (fill_vrt_off[cidx] - voff == 0) return;

    const int32_t i0 = static_cast<int32_t>(cycle_edges[gidx] >> 31);

    int cstart = 0, cend = cycle_off[cidx];
    if (cidx > 0) cstart = cycle_off[cidx - 1];
    const float csize = static_cast<float>(cend - cstart);


    const float3 v = verts[i0];
    atomicAdd(&cycle_center[voff * 3 + 0], v.x / csize);
    atomicAdd(&cycle_center[voff * 3 + 1], v.y / csize);
    atomicAdd(&cycle_center[voff * 3 + 2], v.z / csize);

}


__global__ void make_fill_faces_kernel(
    const int N_CYCLE_EDGES,
    const int N_VERTS,
    const int64_t* __restrict__ cycle_edges,
    const int32_t* __restrict__ cycle_map,
    const int32_t* __restrict__ cycle_off,
    const int32_t* __restrict__ cycle_ori,
    const int32_t* __restrict__ fill_vrt_off,
    const int32_t* __restrict__ fill_tri_off,
    int32_t* __restrict__ extra_faces
){

    const int gidx = global_idx();
    if (gidx >= N_CYCLE_EDGES) return;

    const int64_t edge = cycle_edges[gidx];
    int64_t i0 = edge >> 31, i1 = edge & ((1LL << 31) - 1);

    int cidx = cycle_map[gidx];

    int estart = 0;
    if (cidx > 0) estart = cycle_off[cidx - 1];
    int local_edge_idx = gidx - estart;

    int fstart = 0, fend = fill_tri_off[cidx];
    if (cidx > 0) fstart = fill_tri_off[cidx - 1];
    int fsize = fend - fstart;

    if (fsize == 1){
        extra_faces[fstart * 3 + local_edge_idx] = i0; // each of 3 edges put single vertex
        return;
    }

    int voff = 0;
    if (cidx > 0) voff = fill_vrt_off[cidx - 1];
    voff += N_VERTS; // it gives as single extra vertex idx

    if (cycle_ori[cidx] < 0) {
        int64_t tmp = i0; i0 = i1; i1 = tmp;
    }

    extra_faces[(fstart + local_edge_idx) * 3 + 0] = static_cast<int32_t>(i0);
    extra_faces[(fstart + local_edge_idx) * 3 + 1] = static_cast<int32_t>(i1);
    extra_faces[(fstart + local_edge_idx) * 3 + 2] = static_cast<int32_t>(voff);

}


__global__ void find_cycle_edge_orientation(
    const int N_CYCLE_EDGES,
    const int64_t* __restrict__ cycle_edges,
    const int64_t* __restrict__ sorted_edge_codes,
    const int32_t* __restrict__ edge_code_map,
    const int N_FACES,
    const int32_t* __restrict__ faces,
    const int32_t* __restrict__ cycle_map,
    int32_t* __restrict__ cycle_ori
) {

    const int gidx = global_idx();
    if(gidx >= N_CYCLE_EDGES) return;


    const int64_t edge = cycle_edges[gidx];
    int64_t i0 = edge >> 31, i1 = edge & ((1LL << 31) - 1);

    const int64_t edge_code = (min(i0, i1) << 31) | max(i0, i1);
    int l = 0, r = 3 * N_FACES;
    while(r - l > 1){
        int m = (l + r) >> 1;
        if (sorted_edge_codes[m] <= edge_code) l = m;
        else r = m;
    }

    const int face_idx = edge_code_map[l] % N_FACES;
    int vids[3];
    #pragma unroll
    for(int k = 0; k < 3; k++) vids[k] = faces[face_idx * 3 + k];

    int ori = 1;
    #pragma unroll
    for(int k = 0; k < 3; k++) {
        if (vids[k] == static_cast<int32_t>(i0) && vids[(k + 1) % 3] == static_cast<int32_t>(i1)){
            ori = -1;
        }
    }


    int cidx = cycle_map[gidx];
    atomicAdd(&cycle_ori[cidx], ori);

}

std::vector<torch::Tensor> cuda_fill_holes(torch::Tensor verts, torch::Tensor faces){

    const int threads = 256;
    auto device = faces.device();
    const int N_VERTS = verts.size(0), N_FACES = faces.size(0);

    // make initial int64 edge code by merging two vertices
    auto edge_code = torch::empty({3 * N_FACES}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    build_edge_code_kernel<<<(N_FACES + threads - 1) / threads, threads>>>(
        N_FACES,
        reinterpret_cast<int3*>(faces.data_ptr<int32_t>()),
        edge_code.data_ptr<int64_t>()
    );
    auto _unique_edge_code = torch::_unique2(edge_code, /*sorted=*/true, /*return_inverse=*/true, /*return_counts=*/true);
    auto unique_edge_code = std::get<0>(_unique_edge_code), unique_edge_counts = std::get<2>(_unique_edge_code);

    // keep only edges which occurrence is 1 per faces
    // and compress remained vertices to contiguous space
    auto boundary_edges = unique_edge_code.index({unique_edge_counts.eq(1)});
    if (boundary_edges.size(0) == 0) return {verts, faces};

    auto verts_ids = torch::empty({boundary_edges.size(0) * 2}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    gather_verts_ids_kernel<<<(boundary_edges.size(0) + threads - 1) / threads, threads>>>(
        static_cast<int32_t>(boundary_edges.size(0)),
        boundary_edges.data_ptr<int64_t>(),
        reinterpret_cast<int2*>(verts_ids.data_ptr<int32_t>())
    );
    auto _verts_map = torch::_unique(verts_ids,/*sorted=*/true, /*return_inverse=*/true);
    auto inv_verts_map = std::get<0>(_verts_map).to(torch::kInt32);
    auto verts_map = std::get<1>(_verts_map).to(torch::kInt32);

    // create double sided edges and make links to reverse edge
    auto half_edges = torch::empty({2 * boundary_edges.size(0)}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto reverse_edge = torch::empty({2 * boundary_edges.size(0)}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    build_half_edge_kernel<<<(boundary_edges.size(0) + threads - 1) / threads, threads>>>(
        static_cast<int32_t>(boundary_edges.size(0)),
        reinterpret_cast<int2*>(verts_map.data_ptr<int32_t>()),
        half_edges.data_ptr<int64_t>(),
        reverse_edge.data_ptr<int32_t>()
    );
    auto _half_edges = torch::sort(half_edges);
    half_edges = std::get<0>(_half_edges);

    // repair reverse indexing
    auto half_edges_order = std::get<1>(_half_edges);
    auto inv_half_edges_order = torch::empty_like(half_edges_order);
    inv_half_edges_order.index_put_({half_edges_order}, torch::arange(0, half_edges_order.size(0), torch::TensorOptions().dtype(half_edges_order.dtype()).device(half_edges_order.device())));
    reverse_edge = inv_half_edges_order.index({reverse_edge}).index({half_edges_order}).to(torch::kInt32);

    // count edge number for each remained vertex to build CSR edge array
    const int N_HALF_EDGES = half_edges.size(0);
    const int N_BOUNDARY_VERTS = inv_verts_map.size(0);
    auto voff = torch::zeros({N_BOUNDARY_VERTS}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    build_verts_offset_kernel<<<(N_HALF_EDGES + threads - 1) / threads, threads>>>(
        N_HALF_EDGES,
        half_edges.data_ptr<int64_t>(),
        voff.data_ptr<int32_t>()
    );
    voff.cumsum_(0);

    // build cycles
    auto pair_edge = torch::arange(0, N_HALF_EDGES, torch::TensorOptions().dtype(torch::kInt32).device(device)).bitwise_xor(1);
    torch::Tensor cycle_idx, changed;
    int fix_iters = 0;
    do {

        cycle_idx = torch::arange(0, N_HALF_EDGES, torch::TensorOptions().dtype(torch::kInt32).device(device));

        auto nxt_edge = pair_edge.index({reverse_edge});
        for(size_t step = 1; step < 2 * N_HALF_EDGES; step *= 2){
            cycle_idx = torch::minimum(cycle_idx, cycle_idx.index({nxt_edge}));
            nxt_edge = nxt_edge.index({nxt_edge});
        }

        auto cycle_code = torch::empty({N_HALF_EDGES}, torch::TensorOptions().dtype(torch::kInt64).device(device));
        build_cycle_code_kernel<<<(N_HALF_EDGES + threads - 1) / threads, threads>>>(
            N_HALF_EDGES,
            half_edges.data_ptr<int64_t>(),
            cycle_idx.data_ptr<int32_t>(),
            cycle_code.data_ptr<int64_t>()
        );
        auto _cycle_code = torch::sort(cycle_code);
        cycle_code = std::get<0>(_cycle_code);

        auto cycle_code_order = std::get<1>(_cycle_code).to(torch::kInt32);
        changed = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));

        fix_large_cycle_kernel<<<(N_BOUNDARY_VERTS + threads - 1) / threads, threads>>>(
            N_BOUNDARY_VERTS,
            voff.data_ptr<int32_t>(),
            cycle_idx.data_ptr<int32_t>(),
            cycle_code.data_ptr<int64_t>(),
            cycle_code_order.data_ptr<int32_t>(),
            reverse_edge.data_ptr<int32_t>(),
            pair_edge.data_ptr<int32_t>(),
            changed.data_ptr<int32_t>()
        );


    } while(changed.item<int32_t>() > 0 && fix_iters++ < 100);


    auto main_cycle_idx = torch::minimum(cycle_idx, cycle_idx.index({reverse_edge}));
    auto main_edge_mask = cycle_idx.eq(main_cycle_idx);
    auto cycle_edges = half_edges.index({main_edge_mask});
    cycle_idx = main_cycle_idx;

    const int N_CYCLE_EDGES = cycle_edges.size(0);
    remap_half_edges_kernel<<<(N_CYCLE_EDGES + threads - 1) / threads, threads>>>(
        N_CYCLE_EDGES,
        cycle_edges.data_ptr<int64_t>(),
        inv_verts_map.data_ptr<int32_t>()
    );

    auto _cycle_order = torch::_unique2(cycle_idx.index({main_edge_mask}), /*sorted=*/true, /*return_inverse=*/true, /*return_counts=*/true);
    const int N_CYCLES = std::get<0>(_cycle_order).size(0);
    auto cycle_map = std::get<1>(_cycle_order).to(torch::kInt32);
    auto cycle_offset = std::get<2>(_cycle_order).to(torch::kInt32);
    cycle_offset.cumsum_(0);

    auto cycle_edge_code = torch::empty({N_CYCLE_EDGES}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    build_cycle_edge_code_kernel<<<(N_CYCLE_EDGES + threads - 1) / threads, threads>>>(
        N_CYCLE_EDGES,
        cycle_edges.data_ptr<int64_t>(),
        cycle_map.data_ptr<int32_t>(),
        cycle_edge_code.data_ptr<int64_t>()
    );
    auto cycle_edge_order = torch::argsort(cycle_edge_code);
    cycle_edges = cycle_edges.index({cycle_edge_order});
    cycle_map = cycle_map.index({cycle_edge_order});

    auto cycle_next = torch::empty({N_CYCLE_EDGES}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    find_cycle_next_kernel<<<(N_CYCLE_EDGES + threads - 1) / threads, threads>>>(
        N_CYCLE_EDGES,
        cycle_edges.data_ptr<int64_t>(),
        cycle_map.data_ptr<int32_t>(),
        cycle_offset.data_ptr<int32_t>(),
        cycle_next.data_ptr<int32_t>()
    );

    auto cycle_edges_buf = torch::empty_like(cycle_edges);
    reorder_cycle_edges_kernel<<<(N_CYCLES + threads - 1) / threads, threads>>>(
        N_CYCLES,
        cycle_offset.data_ptr<int32_t>(),
        cycle_edges.data_ptr<int64_t>(),
        cycle_next.data_ptr<int32_t>(),
        cycle_edges_buf.data_ptr<int64_t>()
    );
    std::swap(cycle_edges, cycle_edges_buf);

    auto fill_vrt_cnt = torch::empty({N_CYCLES}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto fill_tri_cnt = torch::empty({N_CYCLES}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    count_fill_triangles_kernel<<<(N_CYCLES + threads - 1) / threads, threads>>>(
        N_CYCLES,
        cycle_offset.data_ptr<int32_t>(),
        fill_tri_cnt.data_ptr<int32_t>(),
        fill_vrt_cnt.data_ptr<int32_t>()
    );
    fill_vrt_cnt.cumsum_(0); fill_tri_cnt.cumsum_(0);

    const int N_EXTRA_VERTS = fill_vrt_cnt.index({N_CYCLES - 1}).item<int32_t>();
    if (N_EXTRA_VERTS > 0){
        auto extra_verts = torch::zeros({N_EXTRA_VERTS, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        make_fill_verts_kernel<<<(N_CYCLE_EDGES +  threads - 1) / threads, threads>>>(
            N_CYCLE_EDGES,
            cycle_edges.data_ptr<int64_t>(),
            cycle_map.data_ptr<int32_t>(),
            cycle_offset.data_ptr<int32_t>(),
            fill_vrt_cnt.data_ptr<int32_t>(),
            reinterpret_cast<float3*>(verts.data_ptr<float>()),
            extra_verts.data_ptr<float>()
        );
        verts = torch::cat({verts, extra_verts});
    }

    auto _sorted_edge_code = torch::sort(edge_code);
    auto sorted_edge_code = std::get<0>(_sorted_edge_code), edge_code_map = std::get<1>(_sorted_edge_code).to(torch::kInt32);
    auto cycle_ori = torch::zeros({N_CYCLES}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    find_cycle_edge_orientation<<<(N_CYCLE_EDGES + threads - 1) / threads, threads>>>(
        N_CYCLE_EDGES,
        cycle_edges.data_ptr<int64_t>(),
        sorted_edge_code.data_ptr<int64_t>(),
        edge_code_map.data_ptr<int32_t>(),
        N_FACES,
        faces.data_ptr<int32_t>(),
        cycle_map.data_ptr<int32_t>(),
        cycle_ori.data_ptr<int32_t>()
    );

    auto extra_faces = torch::empty({fill_tri_cnt.index({N_CYCLES - 1}).item<int32_t>(), 3}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    make_fill_faces_kernel<<<(N_CYCLE_EDGES + threads - 1) / threads, threads>>>(
        N_CYCLE_EDGES,
        N_VERTS,
        cycle_edges.data_ptr<int64_t>(),
        cycle_map.data_ptr<int32_t>(),
        cycle_offset.data_ptr<int32_t>(),
        cycle_ori.data_ptr<int32_t>(),
        fill_vrt_cnt.data_ptr<int32_t>(),
        fill_tri_cnt.data_ptr<int32_t>(),
        extra_faces.data_ptr<int32_t>()
    );


    faces = torch::cat({faces, extra_faces});

    return {verts, faces};
}
