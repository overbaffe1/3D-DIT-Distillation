
#include <torch/extension.h>
#include <mesh.h>

#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat, #x " must be float")
#define CHECK_INT(x) TORCH_CHECK(x.dtype() == torch::kInt32, #x " must be int32")
#define CHECK_INPUT_FLOAT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)
#define CHECK_INPUT_INT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT(x)

void check_tri_inputs(torch::Tensor q, torch::Tensor tri){
    CHECK_INPUT_FLOAT(q); CHECK_INPUT_FLOAT(tri);
    TORCH_CHECK(q.dim() == 2, "q should have 2 dimensions");
    TORCH_CHECK(tri.dim() == 3, "tri should have 3 dimensions (n, 3, 3)");
}


std::vector<torch::Tensor> query_sdf(torch::Tensor q, torch::Tensor tri, bool watertight, int rand_rays, int ext_rays){
    check_tri_inputs(q, tri);
    sdf_out out = cuda_sdf(q, tri, true, true, watertight, rand_rays, ext_rays);
    return {out.sdf, out.tri_idx, out.tri_uv};
}

torch::Tensor query_sign(torch::Tensor q, torch::Tensor tri, bool fast, bool watertight, int rand_rays, int ext_rays){

    check_tri_inputs(q, tri);

    if (fast){
        return cuda_sdf(q, tri, true, false, watertight, rand_rays, ext_rays).sdf;
    }

    torch::Tensor out_sign = torch::empty({(long int) q.size(0)}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));

    cuda_sign(q, tri, out_sign);

    return out_sign;

}

std::vector<torch::Tensor> query_dist(torch::Tensor q, torch::Tensor tri, bool fast){

    check_tri_inputs(q, tri);

    if (fast){
        sdf_out out = cuda_sdf(q, tri, false, true, false, 1, 1);
        return {out.sdf, out.tri_idx, out.tri_uv};
    }

    torch::Tensor out_dist = torch::empty({(long int) q.size(0)}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));

    cuda_dist(q, tri, out_dist);

    return {
        out_dist,
        torch::full({q.size(0)}, -1, torch::TensorOptions().dtype(torch::kInt32).device(q.device())),
        torch::full({q.size(0), 2}, 0.0f, torch::TensorOptions().dtype(torch::kFloat32).device(q.device()))
    };
}

torch::Tensor render_sign(torch::Tensor points, torch::Tensor depth, torch::Tensor w2cs, torch::Tensor fovs){
    CHECK_INPUT_FLOAT(points); CHECK_INPUT_FLOAT(depth); CHECK_INPUT_FLOAT(w2cs); CHECK_INPUT_FLOAT(fovs);
    return cuda_render_sign(points, depth, w2cs, fovs);
}

torch::Tensor project_features(torch::Tensor coords, torch::Tensor features, torch::Tensor w2cs, torch::Tensor fovs){
    CHECK_INPUT_FLOAT(coords); CHECK_INPUT_FLOAT(features); CHECK_INPUT_FLOAT(w2cs); CHECK_INPUT_FLOAT(fovs);
    return cuda_project_features(coords, features, w2cs, fovs);
}

torch::Tensor flow_sdf(torch::Tensor tri, int grid_res){
    CHECK_INPUT_FLOAT(tri);
    return cuda_flow_sdf(tri, grid_res);
}

std::vector<torch::Tensor> up_tsdf(torch::Tensor sdf, float threshold){
    CHECK_INPUT_FLOAT(sdf);
    return cuda_up_tsdf(sdf, threshold);
}

void face_propagate(torch::Tensor labels, torch::Tensor face2face){
    CHECK_INPUT_INT(labels); CHECK_INPUT_INT(face2face);
    TORCH_CHECK(labels.size(0) == face2face.size(0), "number of faces and labels doesn't match");
    cuda_face_propagate(labels, face2face);
}

torch::Tensor get_oriented_components(torch::Tensor verts, torch::Tensor faces, float thr) {
    CHECK_INPUT_INT(faces); CHECK_INPUT_FLOAT(verts);
    TORCH_CHECK(thr >= 0.0f && thr <= 3.141592653589794, "thr must be between 0.0 and pi");
    return cuda_oriented_components(verts, faces, thr);
}

torch::Tensor local_features(torch::Tensor base_pcd, torch::Tensor base_nrm, float dist,  torch::Tensor v, torch::Tensor f, float farea){
    CHECK_INPUT_FLOAT(base_pcd); CHECK_INPUT_FLOAT(base_nrm); CHECK_INPUT_FLOAT(v); CHECK_INPUT_INT(f);
    return cuda_local_features(base_pcd, base_nrm, dist, v, f, farea);
}

torch::Tensor mesh_intersection(torch::Tensor v, torch::Tensor f, const float tol){
    CHECK_INPUT_FLOAT(v); CHECK_INPUT_INT(f);
    return cuda_mesh_intersection(v, f, tol);
}

torch::Tensor voxels_from_triangles(torch::Tensor tri, int grid_res){
    CHECK_INPUT_FLOAT(tri);
    return cuda_voxels_from_triangles(tri, grid_res);
}

torch::Tensor voxels_from_points(torch::Tensor pcd, int grid_res){
    CHECK_INPUT_FLOAT(pcd);
    return cuda_voxels_from_points(pcd, grid_res);
}

torch::Tensor dilate_coords(torch::Tensor coords, int batch_size, int dilate, int grid_res){
    CHECK_INPUT_INT(coords);
    TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 4, "coords must be 4 dimensional");
    return cuda_dilate_coords(coords, batch_size, dilate, grid_res);
}

torch::Tensor find_corner_points(torch::Tensor vidx, torch::Tensor cube_idx, int batch_size, int grid_res) {
    CHECK_INPUT_INT(vidx); CHECK_INPUT_INT(cube_idx);
    TORCH_CHECK(vidx.dim() == 2 && vidx.size(1) == 4, "vidx must be (n, 4) dimensional");
    TORCH_CHECK(cube_idx.dim() == 2 && cube_idx.size(1) == 4, "cube_idx must be (n, 4) dimensional");
    return cuda_find_corner_points(vidx, cube_idx, batch_size, grid_res);
}

torch::Tensor exterior_voxels(torch::Tensor cube_idx, int batch_size, int grid_res) {
    CHECK_INPUT_INT(cube_idx);
    TORCH_CHECK(cube_idx.dim() == 2 && cube_idx.size(1) == 4, "cube_idx must be (n, 4) dimensional");
    return cuda_exterior_voxels(cube_idx, batch_size, grid_res);
}

torch::Tensor mark_corners_from_exterior_voxels(torch::Tensor vidx, torch::Tensor ext_vox, int grid_res) {
    CHECK_INPUT_INT(vidx); CHECK_CUDA(ext_vox); CHECK_CONTIGUOUS(ext_vox);
    TORCH_CHECK(vidx.dim() == 2 && vidx.size(1) == 4, "vidx must be (n, 4) dimensional");
    TORCH_CHECK(ext_vox.dim() == 4 && ext_vox.size(1) == grid_res && ext_vox.size(2) == grid_res && ext_vox.size(3) == grid_res, "ext_vox must be (b, r, r, r)");
    return cuda_mark_corners_from_exterior_voxels(vidx, ext_vox, grid_res);
}


std::vector<torch::Tensor> mesh_simplify(torch::Tensor v, torch::Tensor f, int num_faces, int max_iters){
    CHECK_INPUT_FLOAT(v); CHECK_INPUT_INT(f);
    return cuda_simplify(v, f, num_faces, max_iters);
}

torch::Tensor vert2cube(torch::Tensor v_idx, int grid_res){
    CHECK_INPUT_INT(v_idx);
    TORCH_CHECK(v_idx.dim() == 2 && v_idx.size(1) == 3, "v_idx must be (n, 3) dimensional");
    return cuda_vert2cube(v_idx, grid_res);
}

torch::Tensor orient_rays(torch::Tensor qtri, torch::Tensor tri, int n_rays, int n_bounce){
    CHECK_INPUT_FLOAT(qtri); CHECK_INPUT_FLOAT(tri);
    return cuda_orient_rays(qtri, tri, n_rays, n_bounce);
}

std::vector<torch::Tensor> fill_holes(torch::Tensor verts, torch::Tensor faces){
    CHECK_INPUT_FLOAT(verts); CHECK_INPUT_INT(faces);
    TORCH_CHECK(verts.dim() == 2 && verts.size(1) == 3, "wrong verts shape");
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "wrong verts shape");
    return cuda_fill_holes(verts, faces);
}

torch::Tensor active_voxels(torch::Tensor sdf){
    CHECK_INPUT_FLOAT(sdf);
    TORCH_CHECK(sdf.dim() == 3, "sdf grid must have 3 dimensions");
    return cuda_active_voxels(sdf);
}

torch::Tensor near_voxels(torch::Tensor dist, float thr){
    CHECK_INPUT_FLOAT(dist);
    TORCH_CHECK(dist.dim() == 3, "dist grid must have 3 dimensions");
    return cuda_near_voxels(dist, thr);
}

std::vector<torch::Tensor> vertex_faces(torch::Tensor faces) {
    CHECK_INPUT_INT(faces);
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces should be (N, 3) shape");
    return cuda_vertex_faces(faces);
}

torch::Tensor add_first_item(torch::Tensor item2item, torch::Tensor off)
{

    CHECK_INPUT_INT(item2item); CHECK_INPUT_INT(off);
    TORCH_CHECK(item2item.dim() == 1, "adjacency vector must be 1 dimensional");
    TORCH_CHECK(off.dim() == 1, "offset vector must be 1 dimensional");

    return cuda_add_first_item(item2item, off);
}

std::vector<torch::Tensor> topo_neighbours(torch::Tensor faces){
    CHECK_INPUT_INT(faces);
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces should be (N, 3) shape");
    return cuda_topo_neighbours(faces);
}

std::vector<torch::Tensor> geo_neighbours(torch::Tensor verts, torch::Tensor faces, float radius){
    CHECK_INPUT_FLOAT(verts);
    TORCH_CHECK(verts.dim() == 2 && verts.size(1) == 3, "verts should be (N, 3) shape");
    CHECK_INPUT_INT(faces);
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces should be (M, 3) shape");
    return cuda_geo_neighbours(verts, faces, radius);
}

std::vector<torch::Tensor> guidance_normals(
    torch::Tensor verts, torch::Tensor faces,
    torch::Tensor face2face, torch::Tensor foff
) {
    CHECK_INPUT_FLOAT(verts);
    TORCH_CHECK(verts.dim() == 2 && verts.size(1) == 3, "verts should be (N, 3) shape");
    CHECK_INPUT_INT(faces);
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces should be (M, 3) shape");
    CHECK_INPUT_INT(face2face); CHECK_INPUT_INT(foff);
    TORCH_CHECK(foff.dim() == 1 && foff.size(0) == faces.size(0), "foff and number of faces mut be the same");

    return cuda_guidance_normals(verts, faces, face2face, foff);
}

torch::Tensor neighbour_faces_ori_scores(
    torch::Tensor verts, torch::Tensor faces, torch::Tensor face_normals,
    torch::Tensor face2face, torch::Tensor foff,
    float sigma_dist, float sigma_nrm
) {
    CHECK_INPUT_FLOAT(verts);
    TORCH_CHECK(verts.dim() == 2 && verts.size(1) == 3, "verts should be (N, 3) shape");
    CHECK_INPUT_INT(faces);
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces should be (M, 3) shape");
    CHECK_INPUT_FLOAT(face_normals);
    TORCH_CHECK(face_normals.dim() == 2 && face_normals.size(0) == faces.size(0) && face_normals.size(1) == 3, "face_normals should be (M, 3) shape");
    CHECK_INPUT_INT(face2face);
    TORCH_CHECK(face2face.dim() == 2 && face2face.size(1) == 2, "face2face should be paired for efficiency");

    return cuda_neighbour_faces_ori_scores(verts, faces, face_normals, face2face, foff, sigma_dist, sigma_nrm);
}

std::vector<torch::Tensor> fmc_forward(torch::Tensor vpos, torch::Tensor vsdf, torch::Tensor cube_vidx, float isovalue){

    CHECK_INPUT_FLOAT(vpos); CHECK_INPUT_FLOAT(vsdf); CHECK_INPUT_INT(cube_vidx);
    TORCH_CHECK(vpos.dim() == 2 && vpos.size(1) == 3, "wrong vpos shape");
    TORCH_CHECK(vsdf.dim() == 1 && vpos.size(0) == vsdf.size(0), "vpos and vsdf shape does not match");
    TORCH_CHECK(cube_vidx.dim() == 2 && cube_vidx.size(1) == 8, "wrong cube vidx shape");

    return cuda_fmc_forward(vpos, vsdf, cube_vidx, isovalue);
}

std::vector<torch::Tensor> fmc_backward(
    torch::Tensor grad_verts, torch::Tensor edge_idx,
    torch::Tensor vpos, torch::Tensor vsdf, float isovalue
){

    CHECK_INPUT_FLOAT(grad_verts);
    TORCH_CHECK(grad_verts.dim() == 2 && grad_verts.size(1) == 3, "wrong grad_verts shape");
    TORCH_CHECK(edge_idx.device().is_cuda() && edge_idx.is_contiguous() && edge_idx.dtype() == torch::kInt64, "wrong edge_idx tensor type");
    TORCH_CHECK(edge_idx.dim() == 1 && edge_idx.size(0) == grad_verts.size(0), "wrong edge_idx shape");
    CHECK_INPUT_FLOAT(vpos); CHECK_INPUT_FLOAT(vsdf);
    TORCH_CHECK(vpos.dim() == 2 && vpos.size(1) == 3 && vpos.size(0) == vsdf.size(0) && vsdf.dim() == 1, "wrong vpos/vsdf shape")

    return cuda_fmc_backward(grad_verts, edge_idx, vpos, vsdf, isovalue);
}

std::vector<torch::Tensor> cast_rays(
    torch::Tensor tri,
    torch::Tensor ray_ori, torch::Tensor ray_dir, torch::Tensor coords,
    int N_LAYERS
) {
    CHECK_INPUT_FLOAT(tri); CHECK_INPUT_FLOAT(ray_ori); CHECK_INPUT_FLOAT(ray_dir); CHECK_INPUT_INT(coords);

    TORCH_CHECK(tri.dim() == 3 && tri.size(1) == 3 && tri.size(2) == 3, "wrong triangle shape");
    TORCH_CHECK(ray_ori.dim() == 2 && ray_dir.dim() == 2, "rays must have 2 dims");
    TORCH_CHECK(ray_ori.size(1) == 3, "ray must have last dimension 3");
    TORCH_CHECK(ray_ori.size(0) == ray_dir.size(0) && ray_ori.size(1) == ray_dir.size(1), "ray_ori and ray_dir must have same shape");
    TORCH_CHECK(coords.dim() == 2 && coords.size(0) == ray_ori.size(0) && coords.size(1) == 3, "wrong ray coord shape");
    TORCH_CHECK(N_LAYERS >= 1 && N_LAYERS <= 4, "number of layers must be between 1 and 4");

    cast_out cuda_out = cuda_cast_rays(tri, ray_ori, ray_dir, coords, N_LAYERS);

    return {cuda_out.idx, cuda_out.depth, cuda_out.uv};
}

torch::Tensor cast_rays_bwd(
    torch::Tensor tri,
    torch::Tensor ray_ori, torch::Tensor ray_dir, int N_LAYERS,
    torch::Tensor tri_idx,
    torch::Tensor grad_depth, torch::Tensor grad_uv
) {
    CHECK_INPUT_FLOAT(tri);
    CHECK_INPUT_FLOAT(ray_ori); CHECK_INPUT_FLOAT(ray_dir); CHECK_INPUT_INT(tri_idx);
    CHECK_INPUT_FLOAT(grad_depth); CHECK_INPUT_FLOAT(grad_uv);

    TORCH_CHECK(tri.dim() == 3 && tri.size(1) == 3 && tri.size(2) == 3, "wrong triangle shape");
    TORCH_CHECK(ray_ori.dim() == 2 && ray_dir.dim() == 2, "rays must have 2 dims");
    TORCH_CHECK(ray_ori.size(1) == 3, "ray must have last dimension 3");
    TORCH_CHECK(ray_ori.size(0) == ray_dir.size(0) && ray_ori.size(1) == ray_dir.size(1), "ray_ori and ray_dir must have same shape");
    TORCH_CHECK(tri_idx.dim() == 2 && tri_idx.size(1) == ray_ori.size(0) && tri_idx.size(0) == N_LAYERS, "wrong triangle idx shape");

    TORCH_CHECK(grad_depth.dim() == 2 && grad_depth.size(1) == ray_ori.size(0) && grad_depth.size(0) == N_LAYERS, "wrong grad depth shape");
    TORCH_CHECK(grad_uv.dim() == 3 && grad_uv.size(1) == ray_ori.size(0) && grad_uv.size(0) == N_LAYERS && grad_uv.size(2) == 2, "wrong grad uv shape");

    return cuda_cast_rays_bwd(tri, ray_ori, ray_dir, N_LAYERS, tri_idx, grad_depth, grad_uv);
}

std::vector<torch::Tensor> cast_radius_rays(
    torch::Tensor tri,
    torch::Tensor ray_ori,
    torch::Tensor ray_dir,
    torch::Tensor coords,
    int N_CANDIDATES,
    float radius,
    float sigma
) {
    CHECK_INPUT_FLOAT(tri); CHECK_INPUT_FLOAT(ray_ori); CHECK_INPUT_FLOAT(ray_dir); CHECK_INPUT_INT(coords);

    TORCH_CHECK(tri.dim() == 3 && tri.size(1) == 3 && tri.size(2) == 3, "wrong triangle shape");
    TORCH_CHECK(ray_ori.dim() == 2 && ray_dir.dim() == 2, "rays must have 2 dims");
    TORCH_CHECK(ray_ori.size(1) == 3 && ray_dir.size(1) == 3, "ray must have last dimension 3");
    TORCH_CHECK(N_CANDIDATES >= 1 && N_CANDIDATES <= 32, "candidates must be between 1 and 32");

    radius_out out = cuda_cast_radius_rays(tri, ray_ori, ray_dir, coords, N_CANDIDATES, radius, sigma);
    return {out.idx, out.depth, out.uv, out.alpha, out.hard_mask};
}

torch::Tensor cast_radius_rays_bwd(
    torch::Tensor tri,
    torch::Tensor ray_ori,
    torch::Tensor ray_dir,
    torch::Tensor tri_idx,
    torch::Tensor hard_mask,
    torch::Tensor grad_depth,
    torch::Tensor grad_uv,
    torch::Tensor grad_alpha,
    float sigma
) {
    CHECK_INPUT_FLOAT(tri); CHECK_INPUT_FLOAT(ray_ori); CHECK_INPUT_FLOAT(ray_dir);
    CHECK_INPUT_INT(tri_idx); CHECK_INPUT_INT(hard_mask);
    CHECK_INPUT_FLOAT(grad_depth); CHECK_INPUT_FLOAT(grad_uv); CHECK_INPUT_FLOAT(grad_alpha);

    TORCH_CHECK(tri.dim() == 3 && tri.size(1) == 3 && tri.size(2) == 3, "wrong triangle shape");
    TORCH_CHECK(ray_ori.dim() == 2 && ray_dir.dim() == 2, "rays must have 2 dims");
    TORCH_CHECK(ray_ori.size(1) == 3 && ray_dir.size(1) == 3, "ray must have last dimension 3");

    int N_CANDIDATES = tri_idx.size(0);
    int N_RAYS = ray_ori.size(0);

    TORCH_CHECK(tri_idx.dim() == 2 && tri_idx.size(1) == N_RAYS, "wrong triangle idx shape");
    TORCH_CHECK(hard_mask.dim() == 1 && hard_mask.size(0) == N_RAYS, "wrong hard_mask shape");

    TORCH_CHECK(grad_depth.dim() == 2 && grad_depth.size(1) == N_RAYS && grad_depth.size(0) == N_CANDIDATES, "wrong grad depth shape");
    TORCH_CHECK(grad_uv.dim() == 3 && grad_uv.size(1) == N_RAYS && grad_uv.size(0) == N_CANDIDATES && grad_uv.size(2) == 2, "wrong grad uv shape");
    TORCH_CHECK(grad_alpha.dim() == 2 && grad_alpha.size(1) == N_RAYS && grad_alpha.size(0) == N_CANDIDATES, "wrong grad alpha shape");

    return cuda_cast_radius_rays_bwd(tri, ray_ori, ray_dir, tri_idx, hard_mask, grad_depth, grad_uv, grad_alpha, sigma);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("sign", &query_sign, "sign query");
    m.def("dist", &query_dist, "dist query");
    m.def("sdf", &query_sdf, "sdf query");
    m.def("flow_sdf", &flow_sdf, "grid sdf propagation");
    m.def("render_sign", &render_sign, "sign from depth maps");
    m.def("project_features", &project_features, "project features from view to input 3d coords");
    m.def("local_features", &local_features, "gather local features from mesh");
    m.def("up_tsdf", &up_tsdf, "up tsdf to higher resolution");
    m.def("face_propagate", &face_propagate, "propagate labels from faces");
    m.def("get_oriented_components", &get_oriented_components, "get oriented components in mesh");
    m.def("mesh_intersection", &mesh_intersection, "get all mesh self intersections");
    m.def("voxels_from_triangles", &voxels_from_triangles, "get binary voxel grid from triangle mesh");
    m.def("voxels_from_points", &voxels_from_points, "get binary voxel grid from point cloud");
    m.def("active_voxels", &active_voxels, "get active voxels of sdf corner grid");
    m.def("near_voxels", &near_voxels, "get voxels near to surface");
    m.def("dilate_coords", &dilate_coords, "add shift to non zero coords");
    m.def("find_corner_points", &find_corner_points, "find corner points in voxel grid for marching cubes");
    m.def("exterior_voxels", &exterior_voxels, "mark external voxels using mesh idx voxels");
    m.def("mark_corners_from_exterior_voxels", &mark_corners_from_exterior_voxels, "determine corner sign by exterior voxels");
    m.def("vert2cube", &vert2cube, "get cube index from vertex index");
    m.def("simplify", &mesh_simplify, "apply quadric edge collapse for manifold mesh");
    m.def("orient_rays", &orient_rays, "cast random rays from triangles and count how much survive");
    m.def("fill_holes", &fill_holes, "cuda simple hole filling");
    m.def("fmc_forward", &fmc_forward, "flexible marching cubes forward");
    m.def("fmc_backward", &fmc_backward, "flexible marching cubes backward");
    m.def("vertex_faces", &vertex_faces, "calculate vertex faces sparse array");
    m.def("add_first_item", &add_first_item, "add first index to adjacency pairs");
    m.def("topo_neighbours", &topo_neighbours, "calculate topological faces neighbours sparse array");
    m.def("geo_neighbours", &geo_neighbours, "calculate geo faces neighbours sparse array");
    m.def("guidance_normals", &guidance_normals, "calculate guidance normals for mesh faces");
    m.def("neighbour_faces_ori_scores", &neighbour_faces_ori_scores, "make scores by similar orientation for neighbour faces");
    m.def("cast_rays", &cast_rays, "cast rays to mesh and return it's attributes");
    m.def("cast_rays_bwd", &cast_rays_bwd, "compute backward relative to triangles");
    m.def("cast_radius_rays", &cast_radius_rays, "find best k candidate triangles within radius");
    m.def("cast_radius_rays_bwd", &cast_radius_rays_bwd, "compute backward for radius rays");
}
