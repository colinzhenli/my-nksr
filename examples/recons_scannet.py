# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import nksr
import torch
from torch.nn import functional as F

from pycg import vis, exp
from pathlib import Path
import numpy as np
from common import load_scannet_example
from my_metrics import UnitMeshEvaluator
from skimage.measure import marching_cubes

import open3d as o3d

def compute_centroid(points):
    return np.mean(points, axis=0)

def generate_point_cloud(field, input_xyz):
    # self.model.eval()
    num_steps =  9
    threshold =  0.4
    num_points =  1600000
    # num_points: 20000
    filter_val =  0.01
    device = torch.device("cuda")
    # freeze model parameters
    # for param in self.network.parameters():
    #     param.requires_grad = False

    # sample_num set to 200000
    sample_num = 200000

    # Initialize samples in CUDA device
    samples_cpu = np.zeros((0, 3))

    # Initialize voxel center coordinates
    points = input_xyz

    # Initialize samples and move to CUDA device
    min_range, _ = torch.min(points, axis=0)
    max_range, _ = torch.max(points, axis=0)
    samples = torch.rand(1, sample_num, 3).float().to(device)
    samples *= (max_range.to(device) - min_range.to(device))
    samples += min_range.to(device) # make samples within coords_range
    N = samples.shape[1]  # The number of samples
    samples.requires_grad = True

    i = 0
    while len(samples_cpu) < num_points:
        # print('iteration', i)

        for j in range(num_steps):
            df_pred = torch.abs(field.evaluate_f_bar(samples[0]).unsqueeze(0))
            df_pred.sum().backward(retain_graph=True)
            # gradient = samples.grad.unsqueeze(0).detach()
            gradient = samples.grad.detach()
            samples = samples.detach()
            df_pred = df_pred.detach()
            samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  
            samples = samples.detach()
            samples.requires_grad = True

        # print('finished refinement')

        if not i == 0:
            # Move samples to CPU, detach from computation graph, convert to numpy array, and stack to samples_cpu
            samples_cpu = np.vstack((samples_cpu, samples[df_pred < filter_val].detach().cpu().numpy()))

        samples = samples[df_pred < 0.03].unsqueeze(0)
        indices = torch.randint(samples.shape[1], (1, sample_num))
        samples = samples[[[0, ] * sample_num], indices]
        samples += (threshold / 3) * torch.randn(samples.shape).to(device)  # 3 sigma rule
        samples = samples.detach()
        samples.requires_grad = True

        i += 1
        # print(samples_cpu.shape)

    # for param in self.network.parameters():
    #     param.requires_grad = True

    return samples_cpu
    
def reconstruct_mesh(field, input_xyz):
    grid_dim = 256
    total_voxels = grid_dim ** 3  # 128x128x128

    min_range, _ = torch.min(points, axis=0)
    max_range, _ = torch.max(points, axis=0)
    # min_range -= 0.2
    # max_range += 0.2
    # Create separate grid ranges for x, y, and z
    resolution = 0.04  # 0.02 meters (2 cm)

    # Calculate physical size in each dimension
    physical_size_x = max_range[0] - min_range[0]
    physical_size_y = max_range[1] - min_range[1]
    physical_size_z = max_range[2] - min_range[2]

    # Calculate the number of steps in each dimension
    grid_dim_x = int(torch.round(physical_size_x / resolution))
    grid_dim_y = int(torch.round(physical_size_y / resolution))
    grid_dim_z = int(torch.round(physical_size_z / resolution))
    total_voxels = grid_dim_x * grid_dim_y * grid_dim_z

    grid_range_x = torch.linspace(min_range[0], max_range[0], steps=grid_dim_x)
    grid_range_y = torch.linspace(min_range[1], max_range[1], steps=grid_dim_y)
    grid_range_z = torch.linspace(min_range[2], max_range[2], steps=grid_dim_z)
    print(min_range, max_range)

    # Create uniform grid samples for the cuboid
    grid_x, grid_y, grid_z = torch.meshgrid(grid_range_x, grid_range_y, grid_range_z)
    samples = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(1, total_voxels, 3).float().to(device)

    sdf_res = field.evaluate_f_bar(samples[0])
    sdf = sdf_res

    # Reshape the output to grid shape
    sdf = sdf.reshape(grid_dim_x, grid_dim_y, grid_dim_z).cpu().numpy()
    # udf = torch.clamp(torch.abs(sdf), max=0.4).cpu().numpy()
    vertices, faces, normals, values = marching_cubes(sdf, level=0, spacing=(resolution, resolution, resolution))
    # Create an Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    return mesh

if __name__ == '__main__':
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    std_dev = 0.0
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
    # torch.set_grad_enabled(True)
    # dense_pointcloud = generate_point_cloud(field, sparse_input_xyz)
    # torch.set_grad_enabled(False)

    evaluator = UnitMeshEvaluator(
        n_points=100000,
        metric_names=UnitMeshEvaluator.ESSENTIAL_METRICS)
    onet_samples = None
    eval_dict, translation, scale = evaluator.eval_mesh(nksr_mesh, input_xyz, input_normal, onet_samples=onet_samples)
    # eval_dict, translation, scale = evaluator.eval_pointcloud(dense_pointcloud, input_xyz, sparse_input_xyz)

    print(eval_dict)
    # our_mesh = o3d.io.read_triangle_mesh("/localhome/zla247/projects/data/Visualizations/InterpolatedDecoder_Voxel-0.5_['scene0221_00']_noisy_CD-L1_0.0046_num-10000_mesh.obj")
    # gt_mesh = o3d.io.read_triangle_mesh("../projects/data/Visualizations/gt_mesh.obj")  
    # nksr_mesh.translate(translation)
    # nksr_mesh.scale(scale, center=nksr_mesh.get_center())
    # gt_centroid = compute_centroid(np.asarray(gt_mesh.vertices))
    # our_centroid
    # baseline_translation = mesh_centroid - baseline_centroid
    # gt_translation = mesh_centroid - gt_centroid
    # baseline_mesh.translate(baseline_translation)
    # gt_mesh.translate(gt_translation)
    # our_eval_dict = evaluator.eval_mesh(our_mesh, input_xyz, input_normal, onet_samples=onet_samples)
    # print(our_eval_dict)

    o3d.io.write_triangle_mesh("../projects/data/Visualizations/0nksr_mesh_voxel_0.02.obj", nksr_mesh)

    # eval_res = field.evaluate_f(ref_xyz[ref_xyz_inds], grad=compute_grad)