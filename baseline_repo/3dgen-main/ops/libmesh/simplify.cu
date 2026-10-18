

#include <cutil_math.h>
#include <mesh.h>
#include <chrono>

using namespace torch::indexing;

__device__ void add_quadric(double* quadric, const double3 &n, const double d){
    quadric[0 * 3 + 0] += n.x * n.x, quadric[0 * 3 + 1] += n.x * n.y, quadric[0 * 3 + 2] += n.x * n.z;
    quadric[1 * 3 + 0] += n.y * n.x, quadric[1 * 3 + 1] += n.y * n.y, quadric[1 * 3 + 2] += n.y * n.z;
    quadric[2 * 3 + 0] += n.z * n.x, quadric[2 * 3 + 1] += n.z * n.y, quadric[2 * 3 + 2] += n.z * n.z;
    quadric[3 * 3 + 0] += d * n.x, quadric[3 * 3 + 1] += d * n.y, quadric[3 * 3 + 2] += d * n.z;
    quadric[3 * 3 + 3 + 0] += d * d;
}

__device__ float edge_cost(const double* quadric, const float3 v){
    double cost = quadric[12];
    cost += 2 * (v.x * quadric[9] + v.y * quadric[10] + v.z * quadric[11]);
    cost += v.x * quadric[3 * 0 + 0] * v.x + v.x * quadric[3 * 0 + 1] * v.y + v.x * quadric[3 * 0 + 2] * v.z;
    cost += v.y * quadric[3 * 1 + 0] * v.x + v.y * quadric[3 * 1 + 1] * v.y + v.y * quadric[3 * 1 + 2] * v.z;
    cost += v.z * quadric[3 * 2 + 0] * v.x + v.z * quadric[3 * 2 + 1] * v.y + v.z * quadric[3 * 2 + 2] * v.z;
    return (float) cost;
}


__device__ float3 chols(const double* quadric){

    double det = dot(cross(
        make_double3(quadric[0], quadric[1], quadric[2]),
        make_double3(quadric[3], quadric[4], quadric[5])),
        make_double3(quadric[6], quadric[7], quadric[8])
    );
    if (fabs(det) < 1e-4) return make_float3(-8.0f, -8.0f, -8.0f);

    const int n = 3;
    double L[n * n];
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = i; j < n; j++)
        {
            double sum = quadric[i * n + j];
            for(size_t k = 0; k < i; k++) sum -= L[i * n + k] * L[j * n + k];
            if (i == j){
                if (sum <= 1e-6) return make_float3(-8.0f, -8.0f, -8.0f); // non psd matrix return middle pos
                L[i * n + i] = sqrt(sum);
            } else {
                L[j * n + i] = sum / L[i * n + i];
            }
        }
    }


    // Ly = -q2
    double y[3];
    for (size_t i = 0; i < n; i++)
    {
        double sum = -quadric[3 * 3 + i]; // -q2
        for (size_t k = 0; k < i; k++) sum -= L[i * n + k] * y[k];
        y[i] = sum / L[i * n + i];
    }

    // L^Tx = y
    double x[3];
    for(int i = n - 1; i >= 0; i--){
        double sum = y[i];
        for(int k = i + 1; k < n; k++) sum -= L[k * n + i] * x[k];
        x[i] = sum / L[i * n + i];
    }


    return make_float3((float) x[0], (float) x[1], (float) x[2]);
}

__global__ void setup_faces(
    const int n_faces,
    const int *faces,
    const int n_vertices,
    const float *vertices,
    double *quadrics,
    int64_t *codes,
    int64_t *fcodes
) {


    const int gidx = global_idx();
    if (gidx >= n_faces) return;

    int v[3];
    #pragma unroll
    for(int k = 0; k < 3; k++) v[k] = faces[3 * gidx + k];

    // MAKE QUADRICS
    const double3 v0 = make_double3(vertices[3 * v[0] + 0], vertices[3 * v[0] + 1], vertices[3 * v[0] + 2]);
    const double3 v1 = make_double3(vertices[3 * v[1] + 0], vertices[3 * v[1] + 1], vertices[3 * v[1] + 2]);
    const double3 v2 = make_double3(vertices[3 * v[2] + 0], vertices[3 * v[2] + 1], vertices[3 * v[2] + 2]);

    double3 n = normalize(cross(v1 - v0, v2 - v0));
    const double d = -dot(n, (v0 + v1 + v2) / 3.0f);

    double quadric[3 * 3 + 3 + 1];

    #pragma unroll
    for(size_t k = 0; k < 13; k++) quadric[k] = 0.0;
    add_quadric(quadric, n, d);

    for (size_t k = 0; k < 3; k++)
    {
        #pragma unroll
        for (size_t i = 0; i < 13; i++)
        {
            atomicAdd(quadrics + v[k] * 13 + i, quadric[i]);
        }
    }

    // MAKE EDGE CODES
    #pragma unroll
    for (size_t k = 0; k < 3; k++)
    {
        codes[gidx * 6 + k * 2 + 0] = v[k] * (int64_t) n_vertices + v[(k + 1) % 3];
        codes[gidx * 6 + k * 2 + 1] = v[(k + 1) % 3] * (int64_t) n_vertices + v[k];

    }

    // MAKE FACE CODES
    #pragma unroll
    for (size_t k = 0; k < 3; k++)
    {
        fcodes[3 * gidx + k] = v[k] * (int64_t) n_faces + gidx;
    }


}

__global__ void setup_ranges(
    const int n_vertices,
    const int n_edges,
    const int64_t *codes,
    int *vrange,
    const int n_faces,
    const int64_t *fcodes,
    int *frange
){
    const int gidx = global_idx();
    if(gidx >= n_vertices) return;

    // find left and right bound in edges
    int l = -1, r = n_edges, right_bound = n_edges;
    while(r - l > 1){
        int m = (r + l) >> 1;
        int v = codes[m] / (long) n_vertices;
        if (v < gidx) l = m;
        else if (v == gidx) r = m;
        else r = right_bound = m;
    }
    vrange[2 * gidx + 0] = l + 1;

    l = l + 1, r = right_bound;
    while(r - l > 1){
        int m = (r + l) >> 1;
        int v = codes[m] / (int64_t) n_vertices;
        if (v <= gidx) l = m;
        else r = m;
    }
    vrange[2 * gidx + 1] = l;

    // find left and right bound in faces
    l = -1, r = 3 * n_faces, right_bound = 3 * n_faces;
    while(r - l > 1){
        int m = (r + l) >> 1;
        int v = fcodes[m] / (int64_t) n_faces;
        if (v < gidx) l = m;
        else if (v == gidx) r = m;
        else r = right_bound = m;
    }
    frange[2 * gidx + 0] = l + 1;

    l = l + 1, r = right_bound;
    while(r - l > 1){
        int m = (r + l) >> 1;
        if(fcodes[m] / (int64_t) n_faces <= gidx) l = m;
        else r = m;
    }
    frange[2 * gidx + 1] = l;
}


__global__ void calculate_costs(
    const int n_edges,
    const int64_t *codes,
    const int n_vertices,
    const float *vertices,
    const double *quadrics,
    const int n_faces,
    const int *frange,
    const int64_t *fcodes,
    float *costs,
    float *poses
){
    const int gidx = global_idx();
    if(gidx >= n_edges) return;

    const int64_t code = codes[gidx];
    const int i0 = code / (int64_t) n_vertices;
    const int i1 = code % (int64_t) n_vertices;

    const float3 v0 = make_float3(vertices[3 * i0 + 0], vertices[3 * i0 + 1], vertices[3 * i0 + 2]);
    const float3 v1 = make_float3(vertices[3 * i1 + 0], vertices[3 * i1 + 1], vertices[3 * i1 + 2]);

    double quadric[13];
    #pragma unroll
    for (size_t i = 0; i < 13; i++)
    {
        quadric[i] = quadrics[i0 * 13 + i] + quadrics[i1 * 13 + i];
    }

    // we need to solve q1 @ v = -q2 where q1 is symmetric/psd
    float3 v = chols(quadric); float cost;
    if (length(v - make_float3(-8.0f, -8.0f, -8.0f)) < FLT_EPSILON) {
        float c0 = edge_cost(quadric, v0), c1 = edge_cost(quadric, v1), cmid = edge_cost(quadric, (v0 + v1) / 2.0f);
        if (c0 < c1 && c0 < cmid) {
            v = v0; cost = c0;
        } else if (c1 < c0 && c1 < cmid){
            v = v1; cost = c1;
        } else {
            v = (v0 + v1) / 2.0f; cost = cmid;
        }
    } else cost = edge_cost(quadric, v);

    poses[3 * gidx + 0] = v.x;
    poses[3 * gidx + 1] = v.y;
    poses[3 * gidx + 2] = v.z;

    int start0 = frange[2 * i0 + 0], end0 = frange[2 * i0 + 1];
    int start1 = frange[2 * i1 + 0], end1 = frange[2 * i1 + 1];
    int f1 = start1, intersections = 0;
    for(int f0 = start0; f0 <= end0; f0++){
        int a = fcodes[f0] % (int64_t) n_faces;
        int b = fcodes[f1] % (int64_t) n_faces;
        while(b < a && f1 < end1) b = fcodes[++f1] % (int64_t) n_faces;
        if(a == b) intersections++;
    }
    int valence = (end0 - start0 + 1) + (end1 - start1 + 1) - intersections;
    cost = cost / (abs(valence - 6) + 1);

    costs[gidx] = cost;

}


__global__ void select_edges(
    const int n_edges,
    const int64_t *codes,
    const float *costs,
    const float *poses,
    bool *selected,
    const int n_vertices,
    const float *vertices,
    const int *vrange,
    const int *frange,
    const int64_t *fcodes,
    const int n_faces,
    const int *faces
) {

    const int gidx = global_idx();
    if(gidx >= n_edges) return;

    const int64_t code = codes[gidx];
    const int i0 = code / (int64_t) n_vertices;
    const int i1 = code % (int64_t) n_vertices;

    bool select = i0 < i1;

    // CHECK INTERSECTION OF TWO NEIGHBOURHOODS
    int vstart0 = vrange[i0 * 2 + 0], vend0 = vrange[i0 * 2 + 1];
    int vstart1 = vrange[i1 * 2 + 0], vend1 = vrange[i1 * 2 + 1];

    int v_intersections = 0;
    for (int e0 = vstart0, e1 = vstart1; e0 <= vend0 && e1 <= vend1 && select; e0++){
        int u0 = codes[e0] % (int64_t) n_vertices;
        int u1 = codes[e1] % (int64_t) n_vertices;
        while(u1 < u0 && e1 < vend1) u1 = codes[++e1] % (int64_t) n_vertices;
        if (u0 == u1) v_intersections++;
    }
    if (v_intersections != 2) select = false;

    int fstart0 = frange[2 * i0 + 0], fend0 = frange[2 * i0 + 1];
    int fstart1 = frange[2 * i1 + 0], fend1 = frange[2 * i1 + 1];

    for (int vnum = 0; vnum < 2; vnum++){

        int vidx, start, end;
        if(vnum == 0){
            vidx = i0; start = fstart0; end = fend0;
        } else{
            vidx = i1; start = fstart1; end = fend1;
        }

        for(int ff = start; ff <= end && select; ff++) {

            int fidx = fcodes[ff] % (int64_t) n_faces;

            int v[3]; float3 vpos[3];
            for (size_t k = 0; k < 3; k++) {
                v[k] = faces[3 * fidx + k];
                vpos[k] = make_float3(vertices[v[k] * 3 + 0], vertices[v[k] * 3 + 1], vertices[v[k] * 3 + 2]);
            }

            bool has0 = (v[0] == i0) || (v[1] == i0) || (v[2] == i0);
            bool has1 = (v[0] == i1) || (v[1] == i1) || (v[2] == i1);
            if(has0 && has1) continue;

            float3 ncurr = normalize(cross(normalize(vpos[1] - vpos[0]), normalize(vpos[2] - vpos[0])));
            float acurr = length(cross(vpos[1] - vpos[0], vpos[2] - vpos[0]));

            for (size_t k = 0; k < 3; k++){
                if (v[k] != vidx) continue;
                vpos[k] = make_float3(poses[gidx * 3 + 0], poses[gidx * 3 + 1], poses[gidx * 3 + 2]);
            }
            float3 d0 = normalize(vpos[1] - vpos[0]), d1 = normalize(vpos[2] - vpos[0]);
            float3 nnext = normalize(cross(d0, d1));
            float anext = length(cross(vpos[1] - vpos[0], vpos[2] - vpos[0]));

            if (dot(ncurr, nnext) < 0.15f) select = false; // flip normal
            if (anext < acurr * 1e-3f) select = false; // degenerate area
            if (fabs(dot(d0, d1)) > 0.999f) select = false; // degenerate angle
        }
    }

    selected[gidx] = select;

}


__global__ void select_areas(
    const int n_selected,
    const int64_t *selected_edges,
    const int n_edges,
    const int64_t *codes,
    const float *costs,
    int64_t *marks,
    const int n_vertices,
    const int *vrange
) {

    const int gidx = global_idx();
    if(gidx >= n_selected) return;

    const int eidx = selected_edges[gidx];
    const float edge_cost = costs[eidx];

    const uint64_t mark = (((uint64_t) __float_as_uint(edge_cost)) << 32u) | ((uint64_t) eidx);

    const int64_t code = codes[eidx];
    const int i0 = code / (int64_t) n_vertices;
    const int i1 = code % (int64_t) n_vertices;

    int vstart0 = vrange[i0 * 2 + 0], vend0 = vrange[i0 * 2 + 1];
    int vstart1 = vrange[i1 * 2 + 0], vend1 = vrange[i1 * 2 + 1];

    for (int vnum = 0; vnum < 2; vnum++){

        int start, end;
        if(vnum == 0){
            start = vstart0; end = vend0;
        } else{
            start = vstart1; end = vend1;
        }

        for(int ee = start; ee <= end; ee++){
            int vidx = codes[ee] % (int64_t) n_vertices;

            if(!signbit(edge_cost)){
                atomicMin((long long int*) (marks + vidx), (long long int) mark);
            }else{
                atomicMax((unsigned long long int*) (marks + vidx), (unsigned long long int) mark);
            }

        }
    }

}


__global__ void find_min_edges(
    const int n_selected,
    const int64_t *selected_edges,
    const int n_edges,
    const int64_t *codes,
    const int64_t *marks,
    bool *local_minima,
    const int n_vertices,
    const int *vrange
) {
    const int gidx = global_idx();
    if(gidx >= n_selected) return;

    const int eidx = selected_edges[gidx];

    const int64_t code = codes[eidx];
    const int i0 = code / (int64_t) n_vertices;
    const int i1 = code % (int64_t) n_vertices;

    bool is_minima = true;
    for (int vnum = 0; vnum < 2; vnum++){

        int start, end;
        if(vnum == 0){
            start = vrange[i0 * 2 + 0]; end = vrange[i0 * 2 + 1];
        } else{
            start = vrange[i1 * 2 + 0]; end = vrange[i1 * 2 + 1];
        }

        for(int ee = start; ee <= end && is_minima; ee++){
            int vidx = codes[ee] % (int64_t) n_vertices;
            int min_edge = marks[vidx] & 0x7FFFFFFF;
            if (min_edge != eidx) is_minima = false;
        }
    }

    local_minima[gidx] = is_minima;

}


__global__ void collapse_edges(
    const int n_selected,
    const int64_t *selected_edges,
    const int n_edges,
    const int64_t *codes,
    const float *poses,
    bool *valid_edges,
    const int n_faces,
    const int64_t *fcodes,
    bool *valid_faces,
    const int n_vertices,
    const int *vrange,
    const int *frange,
    int *vlink,
    int *blink,
    float *vertices,
    double *quadrics
) {
    const int gidx = global_idx();
    if (gidx >= n_selected) return;
    const int edge_idx = selected_edges[gidx];


    const int64_t code = codes[edge_idx];
    const int i0 = code / (int64_t) n_vertices;
    const int i1 = code % (int64_t) n_vertices;

    // make link for all structures
    vlink[i1] = i0;
    blink[i0] = i1;


    // update pos and vertex quadric
    #pragma unroll
    for (size_t k = 0; k < 3; k++)
    {
        vertices[3 * i0 + k] = poses[3 * edge_idx + k];
    }

    #pragma unroll
    for (size_t i = 0; i < 13; i++)
    {
        quadrics[i0 * 13 + i] = quadrics[i0 * 13 + i] + quadrics[i1 * 13 + i];
    }


    // invalidate forward edge/backward edge/neighbour edges
    // i.e all edges that will merged after edge collapse
    // acutally only the merged edge and two edges that points to i1 from two intersection points
    int vstart0 = vrange[i1 * 2 + 0], vend0 = vrange[i1 * 2 + 1];
    int intersections = 0;
    for(int e0 = vstart0; e0 <= vend0; e0++){
        int u = codes[e0] % (int64_t) n_vertices;
        int vstart1 = vrange[u * 2 + 0], vend1 = vrange[u * 2 + 1];
        bool touched = (u == i0);
        for (int e1 = vstart1; e1 <= vend1; e1++) {
            int v = codes[e1] % (int64_t) n_vertices;
            if(touched && (v == i1)){
                intersections++;
                valid_edges[e0] = false;
                valid_edges[e1] = false;
                break;
            }
            if(v == i0) {
                touched = true;
            }
        }
    }

    // invalidate merged faces actually only 2 faces that left right of merged edge
    int fstart0 = frange[2 * i0 + 0], fend0 = frange[2 * i0 + 1];
    int fstart1 = frange[2 * i1 + 0], fend1 = frange[2 * i1 + 1];
    intersections = 0;

    for(int f0 = fstart0, f1 = fstart1; f0 <= fend0 && f1 <= fend1; f0++){

        int a = fcodes[f0] % (int64_t) n_faces;
        int b = fcodes[f1] % (int64_t) n_faces;

        while(b < a && f1 < fend1) b = fcodes[++f1] % (int64_t) n_faces;

        if(a == b){
            valid_faces[a] = false;
            intersections++;
        }

    }
}

__global__ void merge_faces(
    const int n_selected_faces,
    const int64_t *selected_faces,
    const int *vlink,
    const int64_t *inv_selected_vertices,
    int* f
){
    const int gidx = global_idx();
    if (gidx >= n_selected_faces) return;

    const int fidx = selected_faces[gidx];

    #pragma unroll
    for(int k = 0; k < 3; k++){
        int v = f[3 * fidx + k];
        while(vlink[v] >= 0) v = vlink[v];
        f[3 * fidx + k] = inv_selected_vertices[v];
    }

}

__global__ void calculate_sizes(
    const int n_vertices,
    const int64_t* inv_selected_vertices,
    const int *vrange,
    const int *vlink,
    const bool *valid_edges,
    int64_t *codes,
    int *valid_size,
    int *vsize,
    const int n_faces,
    const int64_t* inv_selected_faces,
    const int *frange,
    const bool* valid_faces,
    int64_t *fcodes,
    int *valid_fsize,
    int *fsize
){
    const int gidx = global_idx();
    if (gidx >= n_vertices) return;

    int target_idx = gidx;
    while(vlink[target_idx] >= 0) target_idx = vlink[target_idx];
    target_idx = inv_selected_vertices[target_idx];

    // count all edges marked as invalid and put them on top of the list
    int vstart = vrange[2 * gidx + 0], vend = vrange[2 * gidx + 1];
    int v_valid = 0;
    for(int e = vstart; e <= vend; e++){

        bool is_valid = valid_edges[e];
        v_valid += is_valid;
        if(!is_valid) {
            codes[e] = (int64_t) 0x7FFFFFFFFFFFFFFF;
            continue;
        }

        int64_t curr_code = codes[e];
        int64_t u = curr_code / (int64_t) n_vertices, v = curr_code % (int64_t) n_vertices;
        while(vlink[v] >= 0) v = vlink[v];

        curr_code = u * n_vertices + v;
        int prev_idx = e - 1;
        while(prev_idx >= vstart){
            int64_t prev_code = codes[prev_idx];
            if (prev_code > curr_code){
                codes[prev_idx + 1] = prev_code;
                prev_idx--;
            } else break;
        }
        codes[prev_idx + 1] = curr_code;

    }


    valid_size[gidx] = v_valid;
    atomicAdd(vsize + target_idx, v_valid);

     // count all faces marked as invalid and put them on top of the list
    int fstart = frange[2 * gidx + 0], fend = frange[2 * gidx + 1];
    int f_valid = 0;
    for(int fidx = fstart; fidx <= fend; fidx++){

        int f = fcodes[fidx] % (int64_t) n_faces;

        bool is_valid = valid_faces[f];
        f_valid += is_valid;
        if(!is_valid) {
            fcodes[fidx] = (int64_t) 0x7FFFFFFFFFFFFFFF;
            continue;
        }

        int64_t curr_code = gidx * (int64_t) n_faces + f;
        int prev_idx = fidx - 1;
        while(prev_idx >= fstart){
            int64_t prev_code = fcodes[prev_idx];
            if (prev_code > curr_code){
                fcodes[prev_idx + 1] = prev_code;
                prev_idx--;
            } else break;
        }
        fcodes[prev_idx + 1] = curr_code;
    }

    valid_fsize[gidx] = f_valid;
    atomicAdd(fsize + target_idx, f_valid);

}


__global__ void calculate_duplicates(
    const int n_merged,
    const int64_t* merged_vertices,
    const int n_vertices,
    const int64_t* inv_selected_vertices,
    const int *vrange,
    const int *blink,
    int64_t *codes,
    int *valid_size,
    int *vsize
){
    const int gidx = global_idx();
    if (gidx >= n_merged) return;

    const int vidx = merged_vertices[gidx];
    const int merge_idx = blink[vidx];

    int start0 = vrange[2 * vidx + 0];
    int end0 = start0 + valid_size[vidx] - 1;

    int start1 = vrange[2 * merge_idx + 0];
    int end1 = start1 + valid_size[merge_idx] - 1;

    for(int e0 = start0, e1 = start1; e0 <= end0 && e1 <= end1;){
        int u = codes[e0] % (int64_t) n_vertices;
        int v = codes[e1] % (int64_t) n_vertices;
        if(u == v){
            int next_idx = e1 + 1;
            while(next_idx <= end1){
                codes[next_idx - 1] = codes[next_idx];
                next_idx++;
            }
            end1--;
        } else if(v < u){
            e1++;
        } else {
            e0++;
        }

    }

    valid_size[merge_idx] = end1 - start1 + 1;
    vsize[inv_selected_vertices[vidx]]  = (end0 - start0 + 1) + (end1 - start1 + 1);
}


__global__ void merge_edges(
    const int n_selected_vertices,
    const int64_t *selected_vertices,
    const int n_vertices,
    const int64_t *inv_selected_vertices,
    const int *vlink,
    const int *blink,
    const int *vrange,
    const int *valid_size,
    const int *vsize,
    int *selected_vrange,
    const int64_t *codes,
    int64_t *selected_codes,
    const int n_selected_faces,
    const int64_t* inv_selected_faces,
    const int n_faces,
    const int *valid_fsize,
    const int *fsize,
    const int *frange,
    const int64_t *fcodes,
    int *selected_frange,
    int64_t *selected_fcodes
){
    const int gidx = global_idx();
    if (gidx >= n_selected_vertices) return;

    const int vidx = selected_vertices[gidx];

    const int merge_idx = blink[vidx];

    int target_start = gidx > 0 ? vsize[gidx - 1] : 0;
    int target_end = vsize[gidx] - 1;
    selected_vrange[2 * gidx + 0] = target_start;
    selected_vrange[2 * gidx + 1] = target_end;

    int start0 = vrange[2 * vidx + 0];
    int end0 = start0 + valid_size[vidx] - 1;

    int curr_idx = target_start;

    if(merge_idx >= 0){
        int start1 = vrange[2 * merge_idx + 0];
        int end1 = start1 + valid_size[merge_idx] - 1;
        int e1 = start1;
        for(int e0 = start0; e0 <= end0; e0++){
            int u = codes[e0] % (int64_t) n_vertices;
            while(e1 <= end1){
                int v = codes[e1] % (int64_t) n_vertices;
                if (v < u){
                    selected_codes[curr_idx++] = gidx * (int64_t) n_selected_vertices + inv_selected_vertices[v];
                    e1++;
                } else break;
            }
            selected_codes[curr_idx++] = gidx * (int64_t) n_selected_vertices + inv_selected_vertices[u];
        }
        while(e1 <= end1){
            int v = codes[e1++] % (int64_t) n_vertices;
            selected_codes[curr_idx++] = gidx * (int64_t) n_selected_vertices + inv_selected_vertices[v];
        }
    }else{
        for(int e = start0; e <= end0; e++){
            selected_codes[curr_idx++] = gidx * (int64_t) n_selected_vertices + inv_selected_vertices[codes[e] % (int64_t) n_vertices];
        }
    }

    target_start = gidx > 0 ? fsize[gidx - 1] : 0;
    target_end = fsize[gidx] - 1;
    selected_frange[2 * gidx + 0] = target_start;
    selected_frange[2 * gidx + 1] = target_end;

    start0 = frange[2 * vidx + 0];
    end0 = start0 + valid_fsize[vidx] - 1;

    curr_idx = target_start;
    if(merge_idx >= 0){

        int start1 = frange[2 * merge_idx + 0];
        int end1 = start1 + valid_fsize[merge_idx] - 1;
        int f1 = start1;
        for(int f0 = start0; f0 <= end0; f0++){
            int a = fcodes[f0] % (int64_t) n_faces;
            while(f1 <= end1){
                int b = fcodes[f1] % (int64_t) n_faces;
                if (b < a){
                    selected_fcodes[curr_idx++] = gidx * (int64_t) n_selected_faces + inv_selected_faces[b];
                    f1++;
                } else break;
            }
            selected_fcodes[curr_idx++] = gidx * (int64_t) n_selected_faces + inv_selected_faces[a];
        }
        while(f1 <= end1){
            int b = fcodes[f1++] % (int64_t) n_faces;
            selected_fcodes[curr_idx++] = gidx * (int64_t) n_selected_faces + inv_selected_faces[b];
        }
    }else{
        for(int fidx = start0; fidx <= end0; fidx++){
            selected_fcodes[curr_idx++] = gidx * (int64_t) n_selected_faces + inv_selected_faces[fcodes[fidx] % (int64_t) n_faces];
        }
    }

}


std::vector<torch::Tensor> cuda_simplify(torch::Tensor v, torch::Tensor f, int num_faces, int max_iters) {

    const int threads = 128;
    int n_vertices = v.size(0), n_faces = f.size(0);

    // VERTEX QUADRICS AND MAIN DATA STRUCTURES
    auto quadrics = torch::zeros({n_vertices, (3 * 3 + 3 + 1)}, torch::TensorOptions().dtype(torch::kFloat64).device(v.device()));
    auto codes = torch::empty({n_faces * 6}, torch::TensorOptions().dtype(torch::kInt64).device(f.device()));
    auto fcodes = torch::empty({n_faces * 3}, torch::TensorOptions().dtype(torch::kInt64).device(f.device()));

    setup_faces<<<(n_faces + threads - 1) / threads, threads>>>(
        n_faces,
        f.data_ptr<int>(),
        n_vertices,
        v.data_ptr<float>(),
        quadrics.data_ptr<double>(),
        codes.data_ptr<int64_t>(),
        fcodes.data_ptr<int64_t>()
    );

    fcodes = std::get<0>(torch::_unique(fcodes, true /*sorted*/, false /*inverse index*/));
    auto _codes = torch::_unique2(codes, /*sorted=*/true, /*return_inverse=*/false, /*return_counts=*/true);
    codes = std::get<0>(_codes);//.index({std::get<2>(_codes).eq(2)}); // take only manifold edges
    int n_edges = codes.size(0);

    // FIND INDEXES FOR DATA STRUCTURES
    auto vrange = torch::empty(
        {n_vertices, 2}, torch::TensorOptions().dtype(torch::kInt32).device(v.device())
    );
    auto frange = torch::empty(
        {n_vertices, 2}, torch::TensorOptions().dtype(torch::kInt32).device(v.device())
    );

    setup_ranges<<<(n_vertices + threads - 1) / threads, threads>>>(
        n_vertices,
        n_edges,
        codes.data_ptr<int64_t>(),
        vrange.data_ptr<int>(),
        n_faces,
        fcodes.data_ptr<int64_t>(),
        frange.data_ptr<int>()
    );

    int prev_faces = n_faces + 1;
    for(int iter_idx = 0; n_faces > num_faces && prev_faces > n_faces && iter_idx < max_iters; iter_idx++){

        auto poses = torch::empty({n_edges, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(v.device()));
        auto costs = torch::empty({n_edges}, torch::TensorOptions().dtype(torch::kFloat32).device(v.device()));

        // FIND OPTIMAL COST/WEIGHT FOR EACH EDGE
        calculate_costs<<<(n_edges + threads - 1) / threads, threads>>>(
            n_edges,
            codes.data_ptr<int64_t>(),
            n_vertices,
            v.data_ptr<float>(),
            quadrics.data_ptr<double>(),
            n_faces,
            frange.data_ptr<int>(),
            fcodes.data_ptr<int64_t>(),
            costs.data_ptr<float>(),
            poses.data_ptr<float>()
        );


        // BAN NOT SUITABLE EDGES
        auto edges_mask = torch::empty({n_edges}, torch::TensorOptions().dtype(torch::kBool).device(v.device()));
        select_edges<<<(n_edges + threads - 1) / threads, threads>>>(
            n_edges,
            codes.data_ptr<int64_t>(),
            costs.data_ptr<float>(),
            poses.data_ptr<float>(),
            edges_mask.data_ptr<bool>(),
            n_vertices,
            v.data_ptr<float>(),
            vrange.data_ptr<int>(),
            frange.data_ptr<int>(),
            fcodes.data_ptr<int64_t>(),
            n_faces,
            f.data_ptr<int>()
        );

        auto selected_edges = torch::arange(n_edges, torch::TensorOptions().dtype(torch::kInt64).device(v.device())).index({edges_mask});
        if (selected_edges.size(0) < 1) break;
        selected_edges = selected_edges.index({costs.index({selected_edges}).argsort(0 /*dim=0*/, false /*descending=false*/)});
        int max_collapse = (n_faces - num_faces + 1) / 2; // each edge remove 2 faces
        selected_edges = selected_edges.index({Slice(0, max_collapse)});

        int n_selected = selected_edges.size(0);
        if(n_selected < 1) break;

        auto marks = torch::full({n_vertices}, (int64_t) 0x7FFFFFFFFFFFFFFF, torch::TensorOptions().dtype(torch::kInt64).device(v.device()));
        select_areas<<<(n_selected + threads - 1) / threads, threads>>>(
            n_selected,
            selected_edges.data_ptr<int64_t>(),
            n_edges,
            codes.data_ptr<int64_t>(),
            costs.data_ptr<float>(),
            marks.data_ptr<int64_t>(),
            n_vertices,
            vrange.data_ptr<int>()
        );


        // FIND LOCAL MINIMAL EDGES FOR PARALLEL COLLAPSE
        auto minimal_edges = torch::empty({n_selected}, torch::TensorOptions().dtype(torch::kBool).device(v.device()));
        find_min_edges<<<(n_selected + threads - 1) / threads, threads>>>(
            n_selected,
            selected_edges.data_ptr<int64_t>(),
            n_edges,
            codes.data_ptr<int64_t>(),
            marks.data_ptr<int64_t>(),
            minimal_edges.data_ptr<bool>(),
            n_vertices,
            vrange.data_ptr<int>()
        );


        selected_edges = selected_edges.index({minimal_edges});
        n_selected = selected_edges.size(0);
        if(n_selected < 1){
            break;
        }

        // MAIN EDGE COLLAPSE FUNCTION (INVALIDATE EDGES/VERTEX LINKS/UPDATE POSITIONS AND QUADRICS)
        auto valid_edges = torch::full({n_edges}, true, torch::TensorOptions().dtype(torch::kBool).device(v.device()));
        auto valid_faces = torch::full({n_faces}, true, torch::TensorOptions().dtype(torch::kBool).device(v.device()));
        auto vlink = torch::full({n_vertices}, -1, torch::TensorOptions().dtype(torch::kInt32).device(v.device()));
        auto blink = torch::full({n_vertices}, -1, torch::TensorOptions().dtype(torch::kInt32).device(v.device()));

        collapse_edges<<<(n_selected + threads - 1) / threads, threads>>>(
            n_selected,
            selected_edges.data_ptr<int64_t>(),
            n_edges,
            codes.data_ptr<int64_t>(),
            poses.data_ptr<float>(),
            valid_edges.data_ptr<bool>(),
            n_faces,
            fcodes.data_ptr<int64_t>(),
            valid_faces.data_ptr<bool>(),
            n_vertices,
            vrange.data_ptr<int>(),
            frange.data_ptr<int>(),
            vlink.data_ptr<int>(),
            blink.data_ptr<int>(),
            v.data_ptr<float>(),
            quadrics.data_ptr<double>()
        );

        // FIND ALIVED VERTICES
        auto selected_vertices = torch::arange(n_vertices, torch::TensorOptions().dtype(torch::kInt64).device(v.device())).index({vlink < 0});
        int n_selected_vertices = selected_vertices.size(0);
        auto inv_selected_vertices = torch::full({n_vertices}, -1, torch::TensorOptions().dtype(torch::kInt64).device(v.device()));
        inv_selected_vertices.index_put_({selected_vertices}, torch::arange(n_selected_vertices, torch::TensorOptions().dtype(torch::kInt64).device(v.device())));

        // FIND ALIVED FACES
        auto selected_faces = torch::arange(n_faces, torch::TensorOptions().dtype(torch::kInt64).device(f.device())).index({valid_faces});
        int n_selected_faces = selected_faces.size(0);
        auto inv_selected_faces = torch::full({n_faces}, -1, torch::TensorOptions().dtype(torch::kInt64).device(f.device()));
        inv_selected_faces.index_put_({selected_faces}, torch::arange(n_selected_faces, torch::TensorOptions().dtype(torch::kInt64).device(f.device())));

        // SET NEW INDEXES FOR SELECTED FACES
        merge_faces<<<(n_selected_faces + threads - 1) / threads, threads>>>(
            n_selected_faces,
            selected_faces.data_ptr<int64_t>(),
            vlink.data_ptr<int>(),
            inv_selected_vertices.data_ptr<int64_t>(),
            f.data_ptr<int>()
        );

        // RESET DATA STRUCTURES FOR ALIVED VERTICES
        auto valid_size = torch::empty({n_vertices}, torch::TensorOptions().dtype(torch::kInt32).device(v.device()));
        auto vsize = torch::full({n_selected_vertices}, 0, torch::TensorOptions().dtype(torch::kInt32).device(v.device()));
        auto valid_fsize = torch::empty({n_vertices}, torch::TensorOptions().dtype(torch::kInt32).device(f.device()));
        auto fsize = torch::full({n_selected_vertices}, 0, torch::TensorOptions().dtype(torch::kInt32).device(f.device()));

        calculate_sizes<<<(n_vertices + threads - 1) / threads, threads>>>(
            n_vertices,
            inv_selected_vertices.data_ptr<int64_t>(),
            vrange.data_ptr<int>(),
            vlink.data_ptr<int>(),
            valid_edges.data_ptr<bool>(),
            codes.data_ptr<int64_t>(),
            valid_size.data_ptr<int>(),
            vsize.data_ptr<int>(),
            n_faces,
            inv_selected_faces.data_ptr<int64_t>(),
            frange.data_ptr<int>(),
            valid_faces.data_ptr<bool>(),
            fcodes.data_ptr<int64_t>(),
            valid_fsize.data_ptr<int>(),
            fsize.data_ptr<int>()
        );
        fsize.cumsum_(0);


        // some neighbor vertices may duplicate because of parallel reduction (only if not block the full neighbourhood)
        auto merged_vertices = torch::arange(n_vertices, torch::TensorOptions().dtype(torch::kInt64).device(v.device())).index({blink >= 0});
        int n_merged = merged_vertices.size(0);

        calculate_duplicates<<<(n_merged + threads - 1) / threads, threads>>>(
            n_merged,
            merged_vertices.data_ptr<int64_t>(),
            n_vertices,
            inv_selected_vertices.data_ptr<int64_t>(),
            vrange.data_ptr<int>(),
            blink.data_ptr<int>(),
            codes.data_ptr<int64_t>(),
            valid_size.data_ptr<int>(),
            vsize.data_ptr<int>()
        );
        vsize.cumsum_(0);

        auto selected_vrange = torch::empty({n_selected_vertices, 2}, torch::TensorOptions().dtype(torch::kInt32).device(v.device()));
        auto selected_codes = torch::empty({vsize.index({vsize.size(0)-1}).item<int>()}, torch::TensorOptions().dtype(torch::kInt64).device(v.device()));
        auto selected_frange = torch::empty({n_selected_vertices, 2}, torch::TensorOptions().dtype(torch::kInt32).device(f.device()));
        auto selected_fcodes = torch::empty({n_selected_faces * 3}, torch::TensorOptions().dtype(torch::kInt64).device(f.device()));

        merge_edges<<<(n_selected_vertices + threads - 1) / threads, threads>>>(
            n_selected_vertices,
            selected_vertices.data_ptr<int64_t>(),
            n_vertices,
            inv_selected_vertices.data_ptr<int64_t>(),
            vlink.data_ptr<int>(),
            blink.data_ptr<int>(),
            vrange.data_ptr<int>(),
            valid_size.data_ptr<int>(),
            vsize.data_ptr<int>(),
            selected_vrange.data_ptr<int>(),
            codes.data_ptr<int64_t>(),
            selected_codes.data_ptr<int64_t>(),
            n_selected_faces,
            inv_selected_faces.data_ptr<int64_t>(),
            n_faces,
            valid_fsize.data_ptr<int>(),
            fsize.data_ptr<int>(),
            frange.data_ptr<int>(),
            fcodes.data_ptr<int64_t>(),
            selected_frange.data_ptr<int>(),
            selected_fcodes.data_ptr<int64_t>()
        );

        // prepare data for next iteration
        n_vertices = n_selected_vertices;
        v = v.index({selected_vertices});
        quadrics = quadrics.index({selected_vertices});

        prev_faces = n_faces;
        n_faces = n_selected_faces;
        f = f.index({selected_faces});
        frange = selected_frange;
        fcodes = selected_fcodes;

        n_edges = selected_codes.size(0);
        vrange = selected_vrange;
        codes = selected_codes;

    }

    return {v, f};
}