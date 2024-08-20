# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import torch
from abc import ABC
import numpy as np
import time
from typing import Union, Optional
from nksr.svh import SparseFeatureHierarchy
from nksr.meshing import MarchingCubes
from nksr.ext import meshing
from nksr import utils
from sklearn.neighbors import NearestNeighbors
# import torch
# from nksr.svh import SparseFeatureHierarchy
# import torch.nn.functional as F
# import pytorch_lightning as pl
# from pytorch3d.ops import knn_points
# from scipy.spatial import KDTree


class EvaluationResult:
    def __init__(self, value: torch.Tensor, gradient: torch.Tensor = None):
        self.value = value
        self.gradient = gradient

    @classmethod
    def zero(cls, grad: bool = False):
        return EvaluationResult(0, 0 if grad else None)

    def __add__(self, other):
        assert isinstance(other, EvaluationResult)
        return EvaluationResult(
            self.value + other.value,
            (self.gradient + other.gradient) if self.gradient is not None else None
        )

    def __sub__(self, other):
        assert not isinstance(other, EvaluationResult)
        return EvaluationResult(
            self.value - other,
            self.gradient
        )


class MeshingResult:
    def __init__(self, v: torch.Tensor = None, f: torch.Tensor = None, c: torch.Tensor = None):
        self.v = v
        self.f = f
        self.c = c


# class Sampler:
#     def __init__(self, **kwargs):
#         # Default values can be set with kwargs.get('key', default_value)
#         self.voxel_size = kwargs.get('voxel_size')
#         self.adaptive_policy = {
#             'method': 'normal',
#             'tau': 0.1,
#             'depth': 2
#         }
#         self.cfg = kwargs.get('cfg')
#         self.ref_xyz = kwargs.get('ref_xyz')
#         self.ref_normal = kwargs.get('ref_normal')
#         self.svh = self._build_gt_svh()
#         self.kdtree = KDTree(self.ref_xyz.detach().cpu().numpy())

#     def _build_gt_svh(self):
#         gt_svh = SparseFeatureHierarchy(
#             voxel_size=self.voxel_size,
#             depth=self.cfg.svh_tree_depth,
#             device=self.ref_xyz.device
#         )
#         if self.adaptive_policy['method'] == "normal":
#             gt_svh.build_adaptive_normal_variation(
#                 self.ref_xyz, self.ref_normal,
#                 tau=self.adaptive_policy['tau'],
#                 adaptive_depth=self.adaptive_policy['depth']
#             )
#         return gt_svh
    
#     def _get_svh_samples(self, svh, n_samples, expand=0, expand_top=0):
#         """
#         Get random samples, across all layers of the decoder hierarchy
#         :param svh: SparseFeatureHierarchy, hierarchy of spatial features
#         :param n_samples: int, number of total samples
#         :param expand: int, size of expansion
#         :param expand_top: int, size of expansion of the coarsest level.
#         :return: (n_samples, 3) tensor of positions
#         """
#         base_coords, base_scales = [], []
#         for d in range(svh.depth):
#             if svh.grids[d] is None:
#                 continue
#             ijk_coords = svh.grids[d].active_grid_coords()
#             d_expand = expand if d != svh.depth - 1 else expand_top
#             if d_expand >= 3:
#                 mc_offsets = torch.arange(-d_expand // 2 + 1, d_expand // 2 + 1, device=svh.device)
#                 mc_offsets = torch.stack(torch.meshgrid(mc_offsets, mc_offsets, mc_offsets, indexing='ij'), dim=3)
#                 mc_offsets = mc_offsets.view(-1, 3)
#                 ijk_coords = (ijk_coords.unsqueeze(dim=1).repeat(1, mc_offsets.size(0), 1) +
#                               mc_offsets.unsqueeze(0)).view(-1, 3)
#                 ijk_coords = torch.unique(ijk_coords, dim=0)
#             base_coords.append(svh.grids[d].grid_to_world(ijk_coords.float()))
#             base_scales.append(torch.full((ijk_coords.size(0),), svh.grids[d].voxel_size, device=svh.device))
#         base_coords, base_scales = torch.cat(base_coords), torch.cat(base_scales)
#         local_ids = (torch.rand((n_samples,), device=svh.device) * base_coords.size(0)).long()
#         local_coords = (torch.rand((n_samples, 3), device=svh.device) - 0.5) * base_scales[local_ids, None]
#         query_pos = base_coords[local_ids] + local_coords
#         return query_pos

#     def _get_samples(self):
#         all_samples = []
#         for config in self.cfg.samplers:
#             if config.type == "uniform":
#                 all_samples.append(
#                     self._get_svh_samples(self.svh, config.n_samples, config.expand, config.expand_top)
#                 )
#             elif config.type == "band":
#                 band_inds = (torch.rand((config.n_samples, ), device=self.ref_xyz.device) * self.ref_xyz.size(0)).long()
#                 eps = config.eps * self.voxel_size
#                 band_pos = self.ref_xyz[band_inds] + \
#                     self.ref_normal[band_inds] * torch.randn((config.n_samples, 1), device=self.ref_xyz.device) * eps
#                 all_samples.append(band_pos)
#             elif config.type == 'on_surface':
#                 n_subsample = config.subsample
#                 if 0 < n_subsample < self.ref_xyz.size(0):
#                     ref_xyz_inds = (torch.rand((n_subsample,), device=self.ref_xyz.device) *
#                                     self.ref_xyz.size(0)).long()
#                 else:
#                     ref_xyz_inds = torch.arange(self.ref_xyz.size(0), device=self.ref_xyz.device)
#                 all_samples.append(self.ref_xyz[ref_xyz_inds])

#         return torch.cat(all_samples, 0)

#     def transform_field(self, field: torch.Tensor):
#         sdf_config = self.cfg
#         assert sdf_config.gt_type != "binary"
#         truncation_size = sdf_config.gt_band * self.voxel_size
#         if sdf_config.gt_soft:
#             field = torch.tanh(field / truncation_size) * truncation_size
#         else:
#             field = torch.clone(field)
#             field[field > truncation_size] = truncation_size
#             field[field < -truncation_size] = -truncation_size
#         return field

#     # def compute_gt_chi_from_pts(self, query_pos: torch.Tensor):
#     #     mc_query_sdf = -ext.sdfgen.sdf_from_points(query_pos, self.ref_xyz, self.ref_normal, 8, 0.02, False)[0]
#     #     return mc_query_sdf

#     def compute_gt_sdf_from_pts(self, query_pos: torch.Tensor):
#         k = 8  
#         stdv = 0.02
#         # knn_output = knn_points(query_pos.unsqueeze(0), self.ref_xyz.unsqueeze(0), K)
#         # indices = knn_output.idx.squeeze(0)
#         normals = self.ref_normal
#         knn_output = knn_points(query_pos.unsqueeze(0).to(torch.device("cuda")), self.ref_xyz.unsqueeze(0).to(torch.device("cuda")), K=k)
#         indices = knn_output.idx.squeeze(0)
#         # dists, indices = self.kdtree.query(query_pos.detach().cpu().numpy(), k=k)
#         indices = torch.tensor(indices, device=query_pos.device)
#         closest_points = self.ref_xyz[indices]
#         surface_to_queries_vec = query_pos.unsqueeze(1) - closest_points #N, K, 3

#         dot_products = torch.einsum("ijk,ijk->ij", surface_to_queries_vec, normals[indices]) #N, K
#         vec_lengths = torch.norm(surface_to_queries_vec[:, 0, :], dim=-1) 
#         use_dot_product = vec_lengths < stdv
#         sdf = torch.where(use_dot_product, torch.abs(dot_products[:, 0]), vec_lengths)

#         # Adjust the sign of the sdf values based on the majority of dot products
#         num_pos = torch.sum(dot_products > 0, dim=1)
#         inside = num_pos <= (k / 2)
#         sdf[inside] *= -1
        
#         return -sdf
    
class BaseField(ABC):
    """
    Base class for the 3D continuous field:
        f_bar = f - level_set
    """
    def __init__(self, svh: Optional[SparseFeatureHierarchy]):
        self.svh = svh
        self.scale = 1.0
        self.mask_field = None
        self.texture_field = None
        self.set_level_set(0.0)

    def set_level_set(self, level_set: float):
        self.level_set = level_set

    def set_mask_field(self, mask_field: "BaseField"):
        self.mask_field = mask_field

    def set_texture_field(self, texture_field: "BaseField"):
        self.texture_field = texture_field

    def set_scale(self, scale: float):
        self.scale = scale
        if self.mask_field is not None:
            self.mask_field.set_scale(scale)
        if self.texture_field is not None:
            self.texture_field.set_scale(scale)

    def clear_svh_kernel_maps(self):
        if self.svh is not None:
            self.svh.clear_kernel_maps()
        if self.mask_field is not None:
            self.mask_field.clear_svh_kernel_maps()
        if self.texture_field is not None:
            self.texture_field.clear_svh_kernel_maps()

    def to_(self, device: Union[torch.device, str]):
        if self.svh is not None:
            self.svh.to_(device)
        if self.mask_field is not None:
            self.mask_field.to_(device)
        if self.texture_field is not None:
            self.texture_field.to_(device)

    @property
    def device(self):
        return self.svh.device

    def evaluate_f(self, xyz: torch.Tensor, grad: bool = False):
        pass

    def evaluate_f_bar(self, xyz: torch.Tensor, max_points: int = -1, verbose: bool = True):
        n_chunks = int(np.ceil(xyz.size(0) / max_points)) if max_points > 0 else 1
        xyz_chunks = torch.chunk(xyz, n_chunks)
        f_bar_chunks = []

        if verbose and len(xyz_chunks) > 10:
            from tqdm import tqdm
            xyz_chunks = tqdm(xyz_chunks)

        for xyz_chunk in xyz_chunks:
            if self.scale != 1.0:
                xyz_chunk = xyz_chunk / self.scale
            f_chunk = self.evaluate_f(xyz_chunk, grad=True).value
            f_bar_chunks.append(f_chunk - self.level_set)

        return torch.cat(f_bar_chunks)

    def extract_primal_mesh(self, depth: int, resolution: int = 2, trim: bool = True, max_points: int = -1):
        primal_grid = self.svh.grids[depth]
        primal_grid_dense = primal_grid.subdivided_grid(
            resolution,
            torch.ones(primal_grid.num_voxels, dtype=bool, device=self.svh.device))
        dual_grid_dense = primal_grid_dense.dual_grid()

        dual_graph = meshing.primal_cube_graph(primal_grid_dense, dual_grid_dense)
        dual_corner_pos = dual_grid_dense.grid_to_world(dual_grid_dense.active_grid_coords().float())
        if self.scale != 1.0:
            dual_corner_pos = dual_corner_pos * self.scale
        dual_corner_value = self.evaluate_f_bar(dual_corner_pos, max_points=max_points)

        primal_v, primal_f = MarchingCubes().apply(dual_graph, dual_corner_pos, dual_corner_value)

        if self.mask_field is not None and trim:
            vert_mask = self.mask_field.evaluate_f_bar(primal_v, max_points=max_points) < 0.0
            primal_v, primal_f = utils.apply_vertex_mask(primal_v, primal_f, vert_mask)

        if self.texture_field is not None:
            primal_c = self.texture_field.evaluate_f_bar(primal_v, max_points=max_points)
        else:
            primal_c = None

        return MeshingResult(primal_v, primal_f, primal_c)

    def extract_dmc_vertices(self, mise_iter: int = 0, grid_upsample: int = 1,
                          max_depth: int = 100, trim: bool = True, max_points: int = -1):
        """

        Args:
            mise_iter (int): iteration for the Multi-IsoSurface Extraction algorithm
            grid_upsample (int): number of upsample before MISE.
                final grid resolution = actual_voxel_res * grid_upsample * (2 ** mise_iter)
            max_depth:
            trim:
            max_points:

        Returns:

        """
        flattened_grids = []
        for d in range(min(self.svh.depth, max_depth + 1)):
            f_grid = meshing.build_flattened_grid(
                self.svh.grids[d]._grid,
                self.svh.grids[d - 1]._grid if d > 0 else None,
                d != self.svh.depth - 1
            )
            if grid_upsample > 1:
                f_grid = f_grid.subdivided_grid(grid_upsample)
            flattened_grids.append(f_grid)
        dual_grid = meshing.build_joint_dual_grid(flattened_grids)
        dmc_graph = meshing.dual_cube_graph(flattened_grids, dual_grid)
        dmc_vertices = torch.cat([
            f_grid.grid_to_world(f_grid.active_grid_coords().float())
            for f_grid in flattened_grids if f_grid.num_voxels() > 0
        ], dim=0)
        if self.scale != 1.0:
            dmc_vertices = dmc_vertices * self.scale

        return dmc_vertices
        
    def extract_dual_mesh(self, mise_iter: int = 0, grid_upsample: int = 1,
                          max_depth: int = 100, trim: bool = True, max_points: int = -1, input_xyz:torch.Tensor = None, gt_xyz: torch.Tensor = None):
        """

        Args:
            mise_iter (int): iteration for the Multi-IsoSurface Extraction algorithm
            grid_upsample (int): number of upsample before MISE.
                final grid resolution = actual_voxel_res * grid_upsample * (2 ** mise_iter)
            max_depth:
            trim:
            max_points:
            gt_xyz: ground truth xyz for mask

        Returns:

        """
        dmc_time = 0.0
        evaluate_time = 0.0
        grid_time = 0.0
        flattened_grids = []
        self.gt_mask = False
        self.mesh_res_growing = True
        self.regular_grid = False
        grid_time -= time.time()
        if not self.mesh_res_growing:
            nksr_svh = SparseFeatureHierarchy(
                voxel_size=0.02,
                depth=self.svh.depth,
                device= input_xyz.device
            )
            if self.regular_grid:
                resolution = 0.1  # Define the resolution of the grid
                distance_threshold = 1  # Define the distance threshold
                offset = 0.5

                min_xyz = torch.min(input_xyz, dim=0).values
                max_xyz = torch.max(input_xyz, dim=0).values
                # Apply the offset to the min and max coordinates
                min_xyz = min_xyz - offset
                max_xyz = max_xyz + offset
                x_range = torch.arange(min_xyz[0], max_xyz[0], resolution)
                y_range = torch.arange(min_xyz[1], max_xyz[1], resolution)
                z_range = torch.arange(min_xyz[2], max_xyz[2], resolution)
                xx, yy, zz = torch.meshgrid(x_range, y_range, z_range)
                grid_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).to(torch.device("cuda"))
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(input_xyz.detach().cpu().numpy())  # coords is an (N, 3) array
                dist, indx = nn.kneighbors(grid_points.detach().cpu().numpy())  # xyz is an (M, 3) array
                dist = torch.from_numpy(dist).to(grid_points.device).squeeze(-1)
                mask = dist <= distance_threshold
                masked_grid_points = grid_points[mask]
                nksr_svh.build_iterative_coarsening(masked_grid_points)
            else:
                nksr_svh.build_point_splatting(input_xyz)
            for d in range(min(nksr_svh.depth, max_depth + 1)):
                f_grid = meshing.build_flattened_grid(
                    nksr_svh.grids[d]._grid,
                    nksr_svh.grids[d - 1]._grid if d > 0 else None,
                    d != nksr_svh.depth - 1
                )
                if grid_upsample > 1:
                    f_grid = f_grid.subdivided_grid(grid_upsample)
                flattened_grids.append(f_grid)

        else:
            for d in range(min(self.svh.depth, max_depth + 1)):
                f_grid = meshing.build_flattened_grid(
                    self.svh.grids[d]._grid,
                    self.svh.grids[d - 1]._grid if d > 0 else None,
                    d != self.svh.depth - 1
                )
                if grid_upsample > 1:
                    f_grid = f_grid.subdivided_grid(grid_upsample)
                flattened_grids.append(f_grid)

        dual_grid = meshing.build_joint_dual_grid(flattened_grids)
        dmc_graph = meshing.dual_cube_graph(flattened_grids, dual_grid)
        dmc_vertices = torch.cat([
            f_grid.grid_to_world(f_grid.active_grid_coords().float())
            for f_grid in flattened_grids if f_grid.num_voxels() > 0
        ], dim=0)
        del flattened_grids, dual_grid
        grid_time += time.time()

        if self.scale != 1.0:
            dmc_vertices = dmc_vertices * self.scale

        evaluate_time -= time.time()
        dmc_value = self.evaluate_f_bar(dmc_vertices, max_points=max_points)
        evaluate_time += time.time()
        for _ in range(mise_iter):
            cube_sign = dmc_value[dmc_graph] > 0
            cube_mask = ~torch.logical_or(torch.all(cube_sign, dim=1), torch.all(~cube_sign, dim=1))
            dmc_graph = dmc_graph[cube_mask]
            unq, dmc_graph = torch.unique(dmc_graph.view(-1), return_inverse=True)
            dmc_graph = dmc_graph.view(-1, 8)
            dmc_vertices = dmc_vertices[unq]
            dmc_graph, dmc_vertices = utils.subdivide_cube_indices(dmc_graph, dmc_vertices)
            dmc_value = self.evaluate_f_bar(dmc_vertices, max_points=max_points)

        dmc_time -= time.time()
        dual_v, dual_f = MarchingCubes().apply(dmc_graph, dmc_vertices, dmc_value)
        dmc_time += time.time()

        if self.mask_field is not None and trim:
            if self.gt_mask:
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(gt_xyz)  # coords is an (N, 3) array
                dist, indx = nn.kneighbors(dual_v.detach().cpu().numpy())  # xyz is an (M, 3) array
                dist = torch.from_numpy(dist).to(dual_v.device).squeeze(-1)
                vert_mask = dist < 0.04
            else:
                evaluate_time = -time.time()
                vert_mask = self.mask_field.evaluate_f_bar(dual_v, max_points=max_points) < 0.0
                evaluate_time += time.time()

            dmc_time -= time.time()
            dual_v, dual_f = utils.apply_vertex_mask(dual_v, dual_f, vert_mask)
            dmc_time += time.time() 

        if self.texture_field is not None:
            dual_c = self.texture_field.evaluate_f_bar(dual_v, max_points=max_points)
        else:
            dual_c = None
        dmc_time -= time.time()
        mesh_res =  MeshingResult(dual_v, dual_f, dual_c)
        dmc_time += time.time()

        return mesh_res, dmc_time, evaluate_time, grid_time
    
    def compute_objective_function(self, xyz:torch.Tensor = None, gt_normals: torch.Tensor = None):
        self.normals = 'default' # 'Analytical' or 'Numerical' or 'default'
        if self.normals == 'default':
            res = self.evaluate_f(xyz, grad=True)
            pd_sdf = res.value
            pd_normals = res.gradient
            pd_normals = -pd_normals / (torch.linalg.norm(pd_normals, dim=-1, keepdim=True) + 1.0e-6)
    
        elif self.normals == 'Analytical':
            xyz.requires_grad = True
            with torch.enable_grad():
                pd_sdf, *_ = self.evaluate_f_bar(xyz, max_points=-1)
                pd_normals = torch.autograd.grad(pd_sdf, [xyz],
                                                    grad_outputs=torch.ones_like(pd_sdf),
                                                    allow_unused=True)[0]
    
        elif self.normals == 'Numerical':
            interval = 0.01 * 0.02
            grad_value = []
            for offset in [(interval, 0, 0), (0, interval, 0), (0, 0, interval)]:
                offset_tensor = torch.tensor(offset, device=self.device)[None, :]
                res_p = self.evaluate_f_bar(xyz + offset_tensor, max_points=-1)
                res_n = self.evaluate_f_bar(xyz - offset_tensor, max_points=-1)
                grad_value.append((res_p - res_n) / (2 * interval))
            pd_normals = torch.stack(grad_value, dim=1)
            pd_sdf = self.evaluate_f_bar(xyz, max_points=-1)
            
        sdf_error = torch.mean(torch.abs(pd_sdf))
        normal_error = torch.mean(torch.norm(pd_normals - gt_normals, dim=1))

        
        return sdf_error, normal_error        
