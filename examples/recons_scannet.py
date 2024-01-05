# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import nksr
import torch

from pycg import vis, exp
from pathlib import Path
import numpy as np
from common import load_scannet_example
from my_metrics import MeshEvaluator
from skimage.measure import marching_cubes

import open3d as o3d

def compute_centroid(points):
    return np.mean(points, axis=0)

def reconstruct_mesh(field, input_xyz):
    grid_dim = 256
    total_voxels = grid_dim ** 3  # 128x128x128

    # Initialize voxel center coordinates
    # points = data_dict['xyz'].detach()
    # voxel_coords = data_dict['voxel_coords'][:, 1:4]  # M, 3
    # voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0  # compute voxel_center in original coordinate system (torch.tensor)
    # voxel_center = voxel_center.to(device)

    # Compute grid range
    min_range, _ = torch.min(input_xyz, axis=0)
    max_range, _ = torch.max(input_xyz, axis=0)
    grid_range = torch.linspace(min_range.min(), max_range.max(), steps=grid_dim)
    print(min_range, max_range)

    # Create uniform grid samples
    grid_x, grid_y, grid_z = torch.meshgrid(grid_range, grid_range, grid_range)
    samples = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(1, total_voxels, 3).float().to(device)

    sdf_res = field.evaluate_f_bar(samples[0])
    sdf = sdf_res

    # Reshape the output to grid shape
    sdf = sdf.reshape(grid_dim, grid_dim, grid_dim).cpu().numpy()
    # udf = torch.clamp(torch.abs(sdf), max=0.4).cpu().numpy()
    vertices, faces, normals, values = marching_cubes(sdf, level=0, spacing=[1.0/255] * 3)
    # Create an Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    return mesh

if __name__ == '__main__':
    device = torch.device("cuda:0")
    std_dev = 0.1
    scannet_geom = load_scannet_example()
    num_input_points = 10000
    input_xyz = torch.from_numpy(scannet_geom["xyz"]).float().to(device)
    input_normal = torch.from_numpy(scannet_geom["normal"]).float().to(device)

    sample_indices = torch.randperm(input_xyz.shape[0])[:num_input_points]
    sparse_input_xyz = input_xyz[sample_indices]
    sparse_input_normal = input_normal[sample_indices]
    if isinstance(std_dev, (float, int)):
        std_dev = [std_dev] * 3  # Same standard deviation for x, y, z
    noise = torch.from_numpy(np.random.normal(0, std_dev, [num_input_points, 3])).to(device)
    sparse_input_xyz += noise
    sparse_input_normal += noise

    # input_xyz = torch.from_numpy(np.asarray(scannet_geom.points)).float().to(device)
    # input_normal = torch.from_numpy(np.asarray(scannet_geom.normals)).float().to(device)

    reconstructor = nksr.Reconstructor(device)
    field = reconstructor.reconstruct(sparse_input_xyz, sparse_input_normal, voxel_size=0.1)
    mesh_res = field.extract_dual_mesh(mise_iter=2)
    nksr_mesh = vis.mesh(mesh_res.v, mesh_res.f)
    # nksr_mesh = reconstruct_mesh(field, input_xyz)

    evaluator = MeshEvaluator(
        n_points=100000,
        metric_names=MeshEvaluator.ESSENTIAL_METRICS)
    onet_samples = None
    eval_dict, translation, scale = evaluator.eval_mesh(nksr_mesh, input_xyz, input_normal, onet_samples=onet_samples)

    print(eval_dict)
    our_mesh = o3d.io.read_triangle_mesh("/localhome/zla247/projects/data/Visualizations/InterpolatedDecoder_Voxel-0.5_['scene0221_00']_noisy_CD-L1_0.0046_num-10000_mesh.obj")
    gt_mesh = o3d.io.read_triangle_mesh("../projects/data/Visualizations/gt_mesh.obj")  
    nksr_mesh.translate(translation)
    nksr_mesh.scale(scale, center=nksr_mesh.get_center())
    # gt_centroid = compute_centroid(np.asarray(gt_mesh.vertices))
    # our_centroid
    # baseline_translation = mesh_centroid - baseline_centroid
    # gt_translation = mesh_centroid - gt_centroid
    # baseline_mesh.translate(baseline_translation)
    # gt_mesh.translate(gt_translation)
    our_eval_dict = evaluator.eval_mesh(our_mesh, input_xyz, input_normal, onet_samples=onet_samples)
    print(our_eval_dict)

    o3d.io.write_triangle_mesh("../projects/data/Visualizations/0.1_noisy_nksr_mesh_voxel_0.1.obj", nksr_mesh)

    # eval_res = field.evaluate_f(ref_xyz[ref_xyz_inds], grad=compute_grad)