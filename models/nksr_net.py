# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import gc
import random
from typing import Optional

import torch
import torchviz
from torch.nn import functional as F
import numpy as np
import open3d as o3d
from nksr import NKSRNetwork, SparseFeatureHierarchy
from nksr.fields import KernelField, NeuralField, LayerField
from nksr.configs import load_checkpoint_from_url
from skimage.measure import marching_cubes


from pycg import exp, vis

from dataset.base import DatasetSpec as DS, list_collate
from models.base_model import BaseModel
from pycg.isometry import ScaledIsometry


# Cache SVH during training, as backward also needs them.
#   (this is due to the intrusive_ptr in ctx only stores the pointer)

SVH_CACHE = []


class Model(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.network = NKSRNetwork(self.hparams)
        if self.hparams.url:
            ckpt_data = load_checkpoint_from_url(self.hparams.url)
            self.network.load_state_dict(ckpt_data['state_dict'])

    @exp.mem_profile(every=1)
    def forward(self, batch, out: dict):
        input_xyz = batch[DS.INPUT_PC][0]
        assert input_xyz.ndim == 2, "Can only forward single batch."

        if self.hparams.feature == 'normal':
            assert DS.TARGET_NORMAL in batch.keys(), "normal must be provided in this config!"
            feat = batch[DS.TARGET_NORMAL][0]
        elif self.hparams.feature == 'sensor':
            assert DS.INPUT_SENSOR_POS in batch.keys(), "sensor must be provided in this config!"
            view_dir = batch[DS.INPUT_SENSOR_POS][0] - input_xyz
            view_dir = view_dir / (torch.linalg.norm(view_dir, dim=-1, keepdim=True) + 1e-6)
            feat = view_dir
        else:
            feat = None
        out['feat'] = feat

        enc_svh = SparseFeatureHierarchy(
            voxel_size=self.hparams.voxel_size,
            depth=self.hparams.tree_depth,
            device=self.device
        )
        enc_svh.build_point_splatting(input_xyz)

        # Compute density by computing points per voxel.
        if self.hparams.runtime_density:
            q_xyz = torch.unique(torch.div(input_xyz, self.hparams.voxel_size).floor().int(), dim=0)
            density = input_xyz.size(0) / q_xyz.size(0)
            exp.logger.info(f"Density {density}, # pts = {input_xyz.size(0)}")

        if self.hparams.runtime_visualize:
            vis.show_3d([vis.pointcloud(input_xyz, normal=feat)], enc_svh.get_visualization())

        feat = self.network.encoder(input_xyz, feat, enc_svh, 0)
        feat, dec_svh, udf_svh = self.network.unet(
            feat, enc_svh,
            adaptive_depth=self.hparams.adaptive_depth,
            gt_decoder_svh=out.get('gt_svh', None)
        )

        if all([dec_svh.grids[d] is None for d in range(self.hparams.adaptive_depth)]):
            if self.trainer.training or self.trainer.validating:
                # In case training data is corrupted (pd & gt not aligned)...
                exp.logger.warning("Empty grid detected during training/validation.")
                return None

        out.update({'enc_svh': enc_svh, 'dec_svh': dec_svh, 'dec_tmp_svh': udf_svh})
        if self.trainer.training:
            SVH_CACHE.append([enc_svh, dec_svh, udf_svh])

        if self.hparams.geometry == 'kernel':
            output_field = KernelField(
                svh=dec_svh,
                interpolator=self.network.interpolators,
                features=feat.basis_features,
                approx_kernel_grad=False
            )
            if self.hparams.solver_verbose:
                output_field.solver_config['verbose'] = True

            normal_xyz = torch.cat([dec_svh.get_voxel_centers(d) for d in range(self.hparams.adaptive_depth)])
            normal_value = torch.cat([feat.normal_features[d] for d in range(self.hparams.adaptive_depth)])

            normal_weight = self.hparams.solver.normal_weight / normal_xyz.size(0) * \
                (self.hparams.voxel_size ** 2)
            output_field.solve_non_fused(
                pos_xyz=input_xyz,
                normal_xyz=normal_xyz,
                normal_value=-normal_value,
                pos_weight=self.hparams.solver.pos_weight / input_xyz.size(0),
                normal_weight=normal_weight,
                reg_weight=1.0
            )

        elif self.hparams.geometry == 'neural':
            output_field = NeuralField(
                svh=dec_svh,
                decoder=self.network.sdf_decoder,
                features=feat.basis_features
            )

        else:
            raise NotImplementedError

        if self.hparams.udf.enabled:
            mask_field = NeuralField(
                svh=udf_svh,
                decoder=self.network.udf_decoder,
                features=feat.udf_features
            )
            mask_field.set_level_set(2 * self.hparams.voxel_size)
        else:
            mask_field = LayerField(dec_svh, self.hparams.adaptive_depth)
        output_field.set_mask_field(mask_field)

        out.update({
            'structure_features': feat.structure_features,
            'normal_features': feat.normal_features,
            'basis_features': feat.basis_features,
            'field': output_field
        })
        return out

    def on_after_backward(self):
        super().on_after_backward()
        SVH_CACHE.clear()

    def transform_field_visualize(self, field: torch.Tensor):
        spatial_config = self.hparams.supervision.spatial
        if spatial_config.gt_type == "binary":
            return torch.tanh(field)
        else:
            if spatial_config.pd_transform:
                from models.loss import KitchenSinkMetricLoss
                return KitchenSinkMetricLoss.transform_field(self.hparams, field)
            else:
                return field

    def compute_gt_svh(self, batch, out):
        if 'gt_svh' in out.keys():
            return out['gt_svh']

        if DS.GT_GEOMETRY in batch.keys():
            ref_geometry = batch[DS.GT_GEOMETRY][0]
            ref_xyz, ref_normal, _ = ref_geometry.torch_attr()
        else:
            ref_xyz, ref_normal = batch[DS.GT_DENSE_PC][0], batch[DS.GT_DENSE_NORMAL][0]

        gt_svh = SparseFeatureHierarchy(
            voxel_size=self.hparams.voxel_size,
            depth=self.hparams.tree_depth,
            device=self.device
        )

        if self.hparams.adaptive_policy.method == "normal":
            gt_svh.build_adaptive_normal_variation(
                ref_xyz, ref_normal,
                tau=self.hparams.adaptive_policy.tau,
                adaptive_depth=self.hparams.adaptive_depth
            )
        else:
            # Not recommended, removed
            raise NotImplementedError

        out['gt_svh'] = gt_svh
        return gt_svh

    @exp.mem_profile(every=1)
    def compute_loss(self, batch, out, compute_metric: bool):
        loss_dict = exp.TorchLossMeter()
        metric_dict = exp.TorchLossMeter()

        from models.loss import GTSurfaceLoss, SpatialLoss, StructureLoss, UDFLoss, ShapeNetIoUMetric

        SpatialLoss.apply(self.hparams, loss_dict, metric_dict, batch, out, compute_metric)
        GTSurfaceLoss.apply(self.hparams, loss_dict, metric_dict, batch, out, compute_metric)

        self.compute_gt_svh(batch, out)
        StructureLoss.apply(self.hparams, loss_dict, metric_dict, batch, out, compute_metric)

        UDFLoss.apply(self.hparams, loss_dict, metric_dict, batch, out, compute_metric)
        ShapeNetIoUMetric.apply(self.hparams, loss_dict, metric_dict, batch, out, compute_metric)

        return loss_dict, metric_dict

    def log_visualizations(self, batch, out, batch_idx):
        if self.trainer.logger is None:
            return
        with torch.no_grad():
            field = out['field']
            if field is None:
                return

            if not self.hparams.no_mesh_vis:
                mesh_res = field.extract_dual_mesh()
                mesh = vis.mesh(mesh_res.v, mesh_res.f)
                self.log_geometry("pd_mesh", mesh)

    def should_use_pd_structure(self, is_val):
        # In case this returns True:
        #   - The tree generation would completely rely on prediction, so does the supervision signal.
        prob = (self.trainer.global_step - self.hparams.structure_schedule.start_step) / \
               (self.hparams.structure_schedule.end_step - self.hparams.structure_schedule.start_step)
        prob = min(max(prob, 0.0), 1.0)
        if not is_val:
            self.log("pd_struct_prob", prob, prog_bar=True, on_step=True, on_epoch=False)
        return random.random() < prob

    # @exp.mem_profile(every=1)
    def train_val_step(self, batch, batch_idx, is_val):
        if batch_idx % 100 == 0:
            gc.collect()

        out = {'idx': batch_idx}
        if not self.should_use_pd_structure(is_val):
            self.compute_gt_svh(batch, out)

        with exp.pt_profile_named("forward"):
            out = self(batch, out)

        # OOM Guard.
        if out is None:
            return None

        with exp.pt_profile_named("loss"):
            loss_dict, metric_dict = self.compute_loss(batch, out, compute_metric=is_val)

        if not is_val:
            self.log_dict_prefix('train_loss', loss_dict)
            if batch_idx % 200 == 0:
                self.log_visualizations(batch, out, batch_idx)
        else:
            self.log_dict_prefix('val_metric', metric_dict)
            self.log_dict_prefix('val_loss', loss_dict)

        loss_sum = loss_dict.get_sum()
        if is_val and torch.any(torch.isnan(loss_sum)):
            exp.logger.warning("Get nan val loss during validation. Setting to 0.")
            loss_sum = 0
        self.log('val_loss' if is_val else 'train_loss/sum', loss_sum)

        return loss_sum

    def generate_point_cloud(self, field, input_xyz, dmc_vertices):
        # self.model.eval()
        num_steps =  9
        threshold =  0.01
        num_points =  1600000
        # num_points: 20000
        filter_val =  0.01
        device = torch.device("cuda")
        # freeze model parameters
        for param in self.network.parameters():
            param.requires_grad = True

        # sample_num set to 200000
        sample_num = 800000

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
        samples = dmc_vertices.to(device)
        indices = torch.randperm(samples.size(0))[:sample_num]
        samples = samples[indices].unsqueeze(0)
        N = samples.shape[1]  # The number of samples
        samples.requires_grad = True

        i = 0
        while len(samples_cpu) < num_points:
            print('iteration', i)

            for j in range(num_steps):
                sdf_pred = field.evaluate_f(samples[0], grad=True)
                df_pred = torch.abs(sdf_pred.value)
                # Visualize the graph before performing the backward pass
                # graph = torchviz.make_dot(df_pred.sum(), params={'samples': samples[0]})

                # # Render the graph (this will display the graph in Jupyter Notebook or save it as a file)
                # graph.render('computation_graph', format='png')  # This saves the graph as 'computation_graph.png'
                # df_pred.sum().backward(retain_graph=True)
                # gradient = samples.grad.unsqueeze(0).detach()
                # gradient = samples.grad.detach()
                pd_grad = sdf_pred.gradient
                sign_mask = torch.sign(df_pred)
                sign_mask = sign_mask.unsqueeze(-1).expand_as(pd_grad)
                pd_grad= pd_grad * sign_mask
                # pd_grad = -pd_grad / (torch.linalg.norm(pd_grad, dim=-1, keepdim=True) + 1.0e-6)
                # loss_dict.add_loss('gt-surface-normal',
                #                    1.0 - torch.sum(pd_grad * ref_normal[ref_xyz_inds], dim=-1).mean(),
                #                    gt_surface_config.normal)
                samples = samples.detach()
                df_pred = df_pred.unsqueeze(0).detach()
                samples = samples - F.normalize(pd_grad.unsqueeze(0), dim=2) * df_pred.reshape(-1, 1)  
                samples = samples.detach()
                samples.requires_grad = True

            print('finished refinement')

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
            print(samples_cpu.shape)

        for param in self.network.parameters():
            param.requires_grad = True

        return samples_cpu
    
    def reconstruct_mesh(self, field, input_xyz):
        grid_dim = 256
        total_voxels = grid_dim ** 3  # 128x128x128

        min_range, _ = torch.min(input_xyz, axis=0)
        max_range, _ = torch.max(input_xyz, axis=0)
        # min_range -= 0.2
        # max_range += 0.2
        # Create separate grid ranges for x, y, and z
        resolution = 0.02  # 0.02 meters (2 cm)

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
        samples = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(1, total_voxels, 3).float().to(torch.device("cuda:0"))

        sdf_res = field.evaluate_f_bar(samples[0])
        sdf = sdf_res

        # Reshape the output to grid shape
        sdf = sdf.reshape(grid_dim_x, grid_dim_y, grid_dim_z).cpu().numpy()
        # udf = torch.clamp(torch.abs(sdf), max=0.4).cpu().numpy()
        vertices, faces, normals, values = marching_cubes(sdf, level=0.00, spacing=(resolution, resolution, resolution))
        # Create an Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

        return mesh

    def test_step(self, batch, batch_idx):
        test_transform, test_inv_transform = None, None
        if self.hparams.test_transform is not None:
            test_transform = ScaledIsometry.from_matrix(np.array(self.hparams.test_transform))
            test_inv_transform = test_transform.inv()

        self.log('source', batch[DS.SCENE_NAME][0])

        out = {'idx': batch_idx}
        self.transform_batch_input(batch, test_transform)

        if self.hparams.test_use_gt_structure:
            self.compute_gt_svh(batch, out)

        out = self(batch, out)

        # loss_dict, metric_dict = self.compute_loss(batch, out, compute_metric=True)
        # self.log_dict(loss_dict)
        # self.log_dict(metric_dict)

        field = out['field']

        mesh_res = field.extract_dual_mesh(grid_upsample=self.hparams.test_n_upsample)
        dmc_vertices = field.extract_dmc_vertices(grid_upsample=self.hparams.test_n_upsample)

        torch.set_grad_enabled(True)
        dense_pointcloud = self.generate_point_cloud(field, batch[DS.INPUT_PC][0], dmc_vertices)
        torch.set_grad_enabled(False)

        mesh = vis.mesh(mesh_res.v, mesh_res.f)
        mesh = self.reconstruct_mesh(field, batch[DS.INPUT_PC][0])
        self.transform_batch_input(batch, test_inv_transform)
        if test_inv_transform is not None:
            mesh = test_inv_transform @ mesh

        if DS.GT_GEOMETRY in batch.keys():
            ref_geometry = batch[DS.GT_GEOMETRY][0]
            ref_xyz, ref_normal, _ = ref_geometry.torch_attr()
        else:
            ref_geometry = None
            ref_xyz, ref_normal = batch[DS.GT_DENSE_PC][0], batch[DS.GT_DENSE_NORMAL][0]

        if self.hparams.test_print_metrics:
            from metrics import UnitMeshEvaluator

            evaluator = UnitMeshEvaluator(
                n_points=100000,
                metric_names=UnitMeshEvaluator.ESSENTIAL_METRICS)
            onet_samples = None
            if DS.GT_ONET_SAMPLE in batch:
                onet_samples = [
                    batch[DS.GT_ONET_SAMPLE][0][0].cpu().numpy(),
                    batch[DS.GT_ONET_SAMPLE][1][0].cpu().numpy()
                ]
            # eval_dict, translation, scale = evaluator.eval_mesh(mesh, ref_xyz, ref_normal, onet_samples=onet_samples)
            dense_pcd = o3d.geometry.PointCloud()
            dense_pcd.points = o3d.utility.Vector3dVector(dense_pointcloud)
            point_cloud_file = f"../data/Visualizations/NKSR_dense_pcd.ply"
            o3d.io.write_point_cloud(point_cloud_file, dense_pcd)     
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(batch[DS.GT_DENSE_PC][0].cpu().numpy())
            gt_point_cloud_file = f"../data/Visualizations/NKSR_gt_pcd.ply"
            o3d.io.write_point_cloud(gt_point_cloud_file, gt_pcd) 
            eval_dict, translation, scale = evaluator.eval_pointcloud(dense_pointcloud.astype(np.float32), ref_xyz.cpu().numpy())
            self.log_dict(eval_dict)
            exp.logger.info("Metric: " + ", ".join([f"{k} = {v:.4f}" for k, v in eval_dict.items()]))

            # nksr_mesh = mesh
            # our_mesh = o3d.io.read_triangle_mesh("/localhome/zla247/projects/data/Visualizations/InterpolatedDecoder_Voxel-0.5_['scene0221_00']_noisy_CD-L1_0.0046_num-10000_mesh.obj")
            # gt_mesh = o3d.io.read_triangle_mesh("../data/Visualizations/gt_mesh.obj")  
            # nksr_mesh.translate(translation)
            # nksr_mesh.scale(scale, center=nksr_mesh.get_center())
            # # gt_centroid = compute_centroid(np.asarray(gt_mesh.vertices))
            # # our_centroid
            # # baseline_translation = mesh_centroid - baseline_centroid
            # # gt_translation = mesh_centroid - gt_centroid
            # # baseline_mesh.translate(baseline_translation)
            # # gt_mesh.translate(gt_translation)
            # # our_eval_dict = evaluator.eval_mesh(our_mesh, ref_xyz, ref_normal, onet_samples=onet_samples)
            # # print(our_eval_dict)

            o3d.io.write_triangle_mesh("../data/Visualizations/Naive-marchingcube_Normal_nksr_mesh_voxel_0.02.obj", mesh)

        input_pc = batch[DS.INPUT_PC][0]

        if self.record_folder is not None:
            # Record also input for comparison.
            self.test_log_data({
                'input': vis.pointcloud(input_pc, normal=out['feat']),
                'mesh': mesh
            })

        if self.hparams.visualize:
            exp.logger.info(f"Visualizing data {batch[DS.SHAPE_NAME][0]}...")
            scenes = vis.show_3d(
                [vis.pointcloud(input_pc), mesh],
                [vis.pointcloud(ref_xyz, normal=ref_normal)],
                point_size=1, use_new_api=False, show=not self.overfit_logger.working,
                viewport_shading='NORMAL', cam_path=f"../cameras/{self.get_dataset_short_name()}.bin"
            )
            self.overfit_logger.log_overfit_visuals({'scene': scenes[0]})


    @classmethod
    def transform_batch_input(cls, batch, transform: Optional[ScaledIsometry]):
        if transform is None:
            return
        batch[DS.INPUT_PC][0] = transform @ batch[DS.INPUT_PC][0]
        if DS.TARGET_NORMAL in batch:
            batch[DS.TARGET_NORMAL][0] = transform.rotation @ batch[DS.TARGET_NORMAL][0]
        if DS.INPUT_SENSOR_POS in batch:
            batch[DS.INPUT_SENSOR_POS][0] = transform @ batch[DS.INPUT_SENSOR_POS][0]

    def get_dataset_spec(self):
        all_specs = [DS.SCENE_NAME, DS.SHAPE_NAME, DS.INPUT_PC,
                     DS.GT_DENSE_PC, DS.GT_DENSE_NORMAL, DS.GT_ONET_SAMPLE,
                     DS.GT_GEOMETRY]
        if self.hparams.feature == 'normal':
            all_specs.append(DS.TARGET_NORMAL)
        elif self.hparams.feature == 'sensor':
            all_specs.append(DS.INPUT_SENSOR_POS)
        return all_specs

    def get_collate_fn(self):
        return list_collate

    def get_hparams_metrics(self):
        return [('val_loss', True)]
