
#ifndef CUMESH_H
#define CUMESH_H

#include <cuda_runtime.h>
#include <torch/extension.h>

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

struct sdf_out{
    torch::Tensor sdf;
    torch::Tensor tri_idx;
    torch::Tensor tri_uv;
    sdf_out(torch::Tensor sdf, torch::Tensor tri_idx, torch::Tensor tri_uv): sdf(sdf), tri_idx(tri_idx), tri_uv(tri_uv) {}
};

void cuda_sign(torch::Tensor q, torch::Tensor tri, torch::Tensor o);

void cuda_dist(torch::Tensor q, torch::Tensor tri, torch::Tensor d);

sdf_out cuda_sdf(torch::Tensor q, torch::Tensor tri, bool use_sign=true, bool use_dist=true, bool check_parity=true, int rand_rays=100, int ext_rays=5);

torch::Tensor cuda_render_sign(torch::Tensor points, torch::Tensor depth, torch::Tensor w2cs, torch::Tensor fovs);

torch::Tensor cuda_project_features(torch::Tensor coords, torch::Tensor features, torch::Tensor w2cs, torch::Tensor fovs);

torch::Tensor cuda_flow_sdf(torch::Tensor tri, int grid_res);

std::vector<torch::Tensor> cuda_up_tsdf(torch::Tensor sdf, float threshold);

torch::Tensor cuda_local_features(torch::Tensor base_pcd, torch::Tensor base_nrm, float dist, torch::Tensor v, torch::Tensor f, float farea);

void cuda_face_propagate(torch::Tensor labels, torch::Tensor face2face);

torch::Tensor cuda_oriented_components(torch::Tensor verts, torch::Tensor faces, float thr);

torch::Tensor cuda_mesh_intersection(torch::Tensor v, torch::Tensor f, float tol);

torch::Tensor cuda_voxels_from_triangles(torch::Tensor tri, int grid_res);

torch::Tensor cuda_voxels_from_points(torch::Tensor pcd, int grid_res);

torch::Tensor cuda_active_voxels(torch::Tensor sdf);

torch::Tensor cuda_near_voxels(torch::Tensor dist, float thr);

torch::Tensor cuda_dilate_coords(torch::Tensor coords, int batch_size, int dilate, int grid_res);

torch::Tensor cuda_find_corner_points(torch::Tensor vidx, torch::Tensor cube_idx, int batch_size, int grid_res);

torch::Tensor cuda_exterior_voxels(torch::Tensor cube_idx, int batch_size, int grid_res);

torch::Tensor cuda_mark_corners_from_exterior_voxels(torch::Tensor vidx, torch::Tensor ext_vox, int grid_res);

torch::Tensor cuda_vert2cube(torch::Tensor v_idx, int grid_res);

std::vector<torch::Tensor> cuda_simplify(torch::Tensor v, torch::Tensor f, int num_faces, int max_iters);

torch::Tensor cuda_orient_rays(torch::Tensor qtri, torch::Tensor tri, int n_rays, int n_bounce);

std::vector<torch::Tensor> cuda_fill_holes(torch::Tensor verts, torch::Tensor faces);

std::vector<torch::Tensor> cuda_vertex_faces(torch::Tensor faces);

torch::Tensor cuda_add_first_item(torch::Tensor item2item, torch::Tensor off);

std::vector<torch::Tensor> cuda_topo_neighbours(torch::Tensor faces);

std::vector<torch::Tensor> cuda_geo_neighbours(torch::Tensor verts, torch::Tensor faces, float radius);

std::vector<torch::Tensor> cuda_guidance_normals(
    torch::Tensor verts, torch::Tensor faces,
    torch::Tensor face2face, torch::Tensor foff
);

torch::Tensor cuda_neighbour_faces_ori_scores(
    torch::Tensor verts, torch::Tensor faces,
    torch::Tensor face_normals, torch::Tensor face2face, torch::Tensor foff,
    float sigma_dist, float sigma_nrm
);

std::vector<torch::Tensor> cuda_fmc_forward(torch::Tensor vpos, torch::Tensor vsdf, torch::Tensor cube_vidx, float isovalue);

std::vector<torch::Tensor> cuda_fmc_backward(
    torch::Tensor grad_verts, torch::Tensor edge_idx,
    torch::Tensor vpos, torch::Tensor vsdf, float isovalue
);

struct cast_out {
    torch::Tensor idx;
    torch::Tensor depth;
    torch::Tensor uv;
    cast_out(torch::Tensor idx, torch::Tensor depth, torch::Tensor uv): idx(idx), depth(depth), uv(uv) {}
};

cast_out cuda_cast_rays(torch::Tensor tri, torch::Tensor ray_ori, torch::Tensor ray_dir, torch::Tensor coords, int N_LAYERS);

torch::Tensor cuda_cast_rays_bwd(
    torch::Tensor tri,
    torch::Tensor ray_ori,
    torch::Tensor ray_dir, int N_LAYERS,
    torch::Tensor tri_idx,
    torch::Tensor grad_depth, torch::Tensor grad_uv
);

struct radius_out {
    torch::Tensor idx;
    torch::Tensor depth;
    torch::Tensor uv;
    torch::Tensor alpha;
    torch::Tensor hard_mask;
    radius_out(torch::Tensor idx, torch::Tensor depth, torch::Tensor uv, torch::Tensor alpha, torch::Tensor hard_mask)
        : idx(idx), depth(depth), uv(uv), alpha(alpha), hard_mask(hard_mask) {}
};

radius_out cuda_cast_radius_rays(
    torch::Tensor tri,
    torch::Tensor ray_ori,
    torch::Tensor ray_dir,
    torch::Tensor coords,
    int N_CANDIDATES,
    float radius,
    float sigma
);

torch::Tensor cuda_cast_radius_rays_bwd(
    torch::Tensor tri,
    torch::Tensor ray_ori,
    torch::Tensor ray_dir,
    torch::Tensor tri_idx,
    torch::Tensor hard_mask,
    torch::Tensor grad_depth,
    torch::Tensor grad_uv,
    torch::Tensor grad_alpha,
    float sigma
);

#endif
