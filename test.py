# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
This file is part of the Zeus deep learning library.
    test.py is mainly used to test an existing model.
"""

import nksr
import time
import json
import torch
import open3d as o3d
from plyfile import PlyData
from torch.nn import functional as F

from pycg import vis, exp
from pathlib import Path
import numpy as np
from metrics import UnitMeshEvaluator, MeshEvaluator
from torch.utils.data import Dataset
from tqdm import tqdm
from dataset.av_gt_geometry import get_class


import zeus
import bdb
import os

import omegaconf

import importlib
import argparse
from pycg import exp, wdb
import pytorch_lightning as pl
from pathlib import Path


def get_default_parser():
    default_parser = argparse.ArgumentParser(add_help=False)
    default_parser = pl.Trainer.add_argparse_args(default_parser)
    return default_parser

class ScanNetDataset(Dataset):
    def __init__(self, split, partial_input=False, **kwargs):
        self.over_fitting = kwargs.get("over_fitting", False)
        self.uniform_sampling = kwargs.get("uniform_sampling", False)
        self.intake_start = kwargs.get("intake_start", 0)
        self.take = kwargs.get("take", 1)
        self.num_input_points = kwargs.get("num_input_points", 5000)
        self.std_dev = kwargs.get("std_dev", 0.00)

        self.split = 'train' if self.over_fitting else split # use only train set for overfitting
        self.base_path = Path(kwargs.get("base_path", None))

        if self.split == "test":
            with (self.base_path / "metadata" / "scannetv2_test.txt").open() as f:
                self.scenes = [t.strip() for t in f.readlines()]
        elif self.split == "train":
            with (self.base_path / "metadata" / "scannetv2_train.txt").open() as f:
                self.scenes = [t.strip() for t in f.readlines()]
        else:
            with (self.base_path / "metadata" / "scannetv2_val.txt").open() as f:
                self.scenes = [t.strip() for t in f.readlines()]
        
        # self.scenes = self.scenes[:4]
        if self.over_fitting:
            with (self.base_path / "metadata" / "scannetv2_val.txt").open() as f:
                self.scenes = [t.strip() for t in f.readlines()]
                self.split = 'val'
            self.scenes = self.scenes[self.intake_start:self.take+self.intake_start]
        
    def __len__(self):
        return len(self.scenes)

    def _get_item(self, data_id, rng):
        if self.over_fitting:
            scene_name = self.scenes[0]

        data = {}
        scene_path = os.path.join(self.base_path, self.split, f"{scene_name}.pth")
        full_data = torch.load(scene_path)
        full_points = full_data['xyz'].astype(np.float32)
        full_normals = full_data['normal'].astype(np.float32)
        seed_value = 42
        num_points = len(full_points)
        np.random.seed(seed_value)
        if self.num_input_points != -1:
            if not self.uniform_sampling:
                # Number of blocks along each axis
                num_blocks = 2
                total_blocks = num_blocks ** 3
                self.common_difference = 200
                # Calculate block sizes
                block_sizes = (full_points.max(axis=0) - full_points.min(axis=0)) / num_blocks

                # Create the number_per_block array with an arithmetic sequence
                average_points_per_block = self.num_input_points // total_blocks
                number_per_block = np.array([
                    average_points_per_block + (i - total_blocks // 2) * self.common_difference
                    for i in range(total_blocks)
                ])
                
                # Adjust number_per_block to ensure the sum is self.num_input_points
                total_points = np.sum(number_per_block)
                difference = self.num_input_points - total_points
                number_per_block[-1] += difference

                # Sample points from each block
                sample_indices = []
                block_index = 0
                total_chosen_indices = 0
                remaining_points = 0  # Points to be added to the next block
                for i in range(num_blocks):
                    for j in range(num_blocks):
                        for k in range(num_blocks):
                            block_min = full_points.min(axis=0) + block_sizes * np.array([i, j, k])
                            block_max = block_min + block_sizes
                            block_mask = np.all((full_points >= block_min) & (full_points < block_max), axis=1)
                            block_indices = np.where(block_mask)[0]
                            num_samples = number_per_block[block_index] + remaining_points
                            remaining_points = 0  # Reset remaining points
                            block_index += 1
                            if len(block_indices) > 0:
                                chosen_indices = np.random.choice(block_indices, num_samples, replace=True)
                                sample_indices.extend(chosen_indices)
                                total_chosen_indices += len(chosen_indices)
                                # print(f"Block {block_index} - Desired: {num_samples}, Actual: {len(chosen_indices)}")
                                if len(chosen_indices) < num_samples:
                                    remaining_points += (num_samples - len(chosen_indices))
                            else:
                                # print(f"Block {block_index} - No points available. Adding {num_samples} points to the next block.")
                                remaining_points += num_samples
            else:
                if num_points < self.num_input_points:
                    print(f"Scene {scene_name} has less than {self.num_input_points} points. Sampling with replacement.")
                    sample_indices = np.random.choice(num_points, self.num_input_points, replace=True)
                else:
                    sample_indices = np.random.choice(num_points, self.num_input_points, replace=True)
            partial_points = full_points[sample_indices]
            partial_normals = full_normals[sample_indices]

        else:
            partial_points = full_points
            partial_normals = full_normals

        if isinstance(self.std_dev, (float, int)):
            std_dev = [self.std_dev] * 3  # Same standard deviation for x, y, z
        noise = np.random.normal(0, self.std_dev, partial_points.shape)
        partial_points += noise

        data = {
            "partial_input": partial_points,
            "partial_normal": partial_normals,
            "full_input": full_points,
            "full_normal": full_normals
        }

        return data
class CarlaDataset(Dataset):
    def __init__(self, split, partial_input=False, **kwargs):
        self.over_fitting = kwargs.get("over_fitting", False)
        self.std_dev = kwargs.get("std_dev", 0.00)
        self.custom_name = 'Carla'
        self.gt_type = "PointTSDFVolume"

        self.split = 'train' if self.over_fitting else split # use only train set for overfitting
        self.file_name = 'pointcloud'
        self.multi_files = 10
        categories = kwargs.get("classes", None)
        self.over_fitting = kwargs.get("over_fitting", False)
        self.intake_start = kwargs.get("intake_start", 1)
        self.take = kwargs.get("take", 1)
        self.num_input_points = kwargs.get("num_input_points", 10000)
        self.std_dev = kwargs.get("std_dev", 0.00)
        self.std_dev *= 2


        self.split = split
        self.use_dummy_gt = False

        # If drives not specified, use all sub-folders
        drives =  ['Town01-0', 'Town01-1', 'Town01-2',
                'Town02-0', 'Town02-1', 'Town02-2',
                'Town10-0', 'Town10-1', 'Town10-2', 'Town10-3', 'Town10-4']

        base_path = '/localhome/zla247/theia2_data/carla-lidar/dataset-no-patch'
        base_path = Path(base_path)
        if drives is None:
            drives = os.listdir(base_path)
            drives = [c for c in drives if (base_path / c).is_dir()]
        self.drives = drives
        self.input_path = '/localhome/zla247/theia2_data/carla-lidar/dataset-p1n2-no-patch'

        # Get all items
        self.all_items = []
        self.drive_base_paths = {}
        for c in drives:
            self.drive_base_paths[c] = base_path / c
            split_file = self.drive_base_paths[c] / (split + '.lst')
            with split_file.open('r') as f:
                models_c = f.read().split('\n')
            if '' in models_c:
                models_c.remove('')
            self.all_items += [{'drive': c, 'item': m} for m in models_c]

        if self.over_fitting:
            self.all_items = self.all_items[self.intake_start:self.take+self.intake_start]

    def __len__(self):
        return len(self.all_items)

    def get_name(self):
        return f"{self.custom_name}-cat{len(self.drives)}-{self.split}"

    def get_short_name(self):
        return self.custom_name

    def _get_item(self, data_id, rng):
        drive_name = self.all_items[data_id]['drive']
        item_name = self.all_items[data_id]['item']

        named_data = {}
        data = {}
        

        try:
            if self.input_path is None:
                input_data = np.load(self.drive_base_paths[drive_name] / item_name / 'pointcloud.npz')
            else:
                input_data = np.load(Path(self.input_path) / drive_name / item_name / 'pointcloud.npz')
        except FileNotFoundError:
            exp.logger.warning(f"File not found for AV dataset for {item_name}")
            raise ConnectionAbortedError
        
        xyz = input_data['points'].astype(np.float32)
        normals = input_data['normals'].astype(np.float32)

        geom_cls = get_class(self.gt_type)
        gt_geometry = geom_cls.load(self.drive_base_paths[drive_name] / item_name / "groundtruth.bin")

        ref_xyz, ref_normal, _ = gt_geometry.torch_attr()


        data = {
            "partial_input": xyz,
            "partial_normal": normals,
            "full_input": ref_xyz.cpu().numpy(),
            "full_normal": ref_normal.cpu().numpy()
        }
        return data

class SyntheticRoomDataset(Dataset):
    def __init__(self, split, partial_input=False, **kwargs):
        self.over_fitting = kwargs.get("over_fitting", False)
        self.std_dev = kwargs.get("std_dev", 0.00)
        self.custom_name = 'Synthetic'
        self.scale = 2.2

        self.split = 'val' if self.over_fitting else split # use only train set for overfitting
        split = self.split
        self.dataset_folder = kwargs.get("base_path", None)
        self.file_name = 'pointcloud'
        self.multi_files = 10
        categories = ['rooms_04', 'rooms_05', 'rooms_06', 'rooms_07', 'rooms_08']
        self.over_fitting = kwargs.get("over_fitting", False)
        self.intake_start = kwargs.get("intake_start", 0)
        self.take = kwargs.get("take", 1)
        self.num_input_points = kwargs.get("num_input_points", 10000)
        self.std_dev = kwargs.get("std_dev", 0.00)
        self.std_dev *= 2

        # self.split = 'val'
        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(self.dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(self.dataset_folder, c))]

        self.metadata = {
            c: {'id': c, 'name': 'n/a'} for c in categories
        } 
        
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.dataset_folder, c)
            if not os.path.isdir(subpath):
                print('Category %s does not exist in dataset.' % c)

            if split is None:
                self.models += [
                    {'category': c, 'model': m} for m in [d for d in os.listdir(subpath) if (os.path.isdir(os.path.join(subpath, d)) and d != '') ]
                ]

            else:
                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
                
                if '' in models_c:
                    models_c.remove('')

                self.models += [
                    {'category': c, 'model': m}
                    for m in models_c
                ]
        
        # overfit in one data
        if self.over_fitting:
            self.models = self.models[self.intake_start:self.take+self.intake_start]


    def __len__(self):
        return len(self.models)

    def get_name(self):
        return f"{self.custom_name}-{self.split}"

    def get_short_name(self):
        return f"{self.custom_name}"
    
    def load(self, model_path, idx, vol):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))
        
        item_path = os.path.join(model_path, 'item_dict.npz')
        item_dict = np.load(item_path, allow_pickle=True)
        points_dict = np.load(file_path, allow_pickle=True)
        points = points_dict['points'] * self.scale # roughly transfer back to physical scale
        normals = points_dict['normals']
        semantics = points_dict['semantics']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            normals = normals.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
            normals += 1e-4 * np.random.randn(*normals.shape)


        # # Flip the y and z axes for points and normals and move to positive quadrant
        # points = points[:, [0, 2, 1]]
        # normals = normals[:, [0, 2, 1]]
        min_values = np.min(points, axis=0)
        points -= min_values

        return {'xyz': points, 'normal': normals, 'semantics': semantics}

    def _get_item(self, data_id, rng):
        data = {}
        category = self.models[data_id]['category']
        model = self.models[data_id]['model']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        full_data = self.load(model_path, data_id, c_idx)
        scene_name = f"{category}/{model}/{data_id}"

        full_points = full_data['xyz'].astype(np.float32)
        full_normals = full_data['normal'].astype(np.float32)

        if self.num_input_points != -1:
            sample_indices = np.random.choice(full_points.shape[0], self.num_input_points, replace=True)
            partial_points = full_points[sample_indices]
            partial_normals = full_normals[sample_indices]

        else:
            partial_points = full_points
            partial_normals = full_normals

        if isinstance(self.std_dev, (float, int)):
            std_dev = [self.std_dev] * 3  # Same standard deviation for x, y, z
        noise = np.random.normal(0, self.std_dev, partial_points.shape)
        partial_points += noise

        data = {
            "partial_input": partial_points,
            "partial_normal": partial_normals,
            "full_input": full_points,
            "full_normal": full_normals
        }

        return data
    
class SceneNNDataset(Dataset):
    def __init__(self, split, partial_input=False, **kwargs):
        self.over_fitting = kwargs.get("over_fitting", False)
        self.num_input_points = kwargs.get("num_input_points", 10000)
        self.std_dev = kwargs.get("std_dev", 0.00)
        self.split = 'train' if self.over_fitting else split # use only train set for overfitting
        self.dataset_folder = Path(kwargs.get("dataset_folder", None))

        self.split = 'val' if self.over_fitting else split # use only train set for overfitting

        self.scenes = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(self.dataset_folder) for f in filenames if f.endswith('.ply')])
        if self.over_fitting:
            self.scenes = [self.scenes[0]]


    def __len__(self):
        return len(self.scenes)

    def _get_item(self, data_id, rng):
        data = {}
        scene_filename = self.scenes[data_id]

        ply_data = PlyData.read(scene_filename)
        vertex = ply_data['vertex']
        pos = np.stack([vertex[t] for t in ('x', 'y', 'z')], axis=1)
        nls = np.stack([vertex[t] for t in ('nx', 'ny', 'nz')], axis=1) if 'nx' in vertex and 'ny' in vertex and 'nz' in vertex else np.zeros_like(pos)

        # if len(pos) > 200000:
        #     indices = np.random.choice(len(pos), 200000, replace=False)
        #     pos = pos[indices]
        #     nls = nls[indices]
        scene_name = os.path.basename(scene_filename).replace('.ply', '')


        full_points = pos
        full_normals = nls

        if self.num_input_points != -1:
            sample_indices = np.random.choice(full_points.shape[0], self.num_input_points, replace=True)
            partial_points = full_points[sample_indices]
            partial_normals = full_normals[sample_indices]

        else:
            partial_points = full_points
            partial_normals = full_normals

        if isinstance(self.std_dev, (float, int)):
            std_dev = [self.std_dev] * 3  # Same standard deviation for x, y, z
        noise = np.random.normal(0, self.std_dev, partial_points.shape)
        partial_points += noise

        data = {
            "partial_input": partial_points,
            "partial_normal": partial_normals,
            "full_input": full_points,
            "full_normal": full_normals
        }

        return data
    
def load_scannet_example():
    scannet_path = Path(__file__).parent.parent / "assets" / "scannet.ply"
    scannet_path = "/localhome/zla247/data/scannetv2/val/scene0221_00.pth"
    scene = torch.load(scannet_path)

    # if not scannet_path.exists():
    #     exp.logger.info("Downloading assets...")
    #     res = requests.get(f"{DOWNLOAD_URL}/scannet-rgbd.ply")
    #     with open(scannet_path, "wb") as f:
    #         f.write(res.content)
    #     exp.logger.info("Download finished!")

    # scannet_geom = vis.from_file(scannet_path)
    # return scannet_geom
    return scene

def convert_non_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.int32):
        return int(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')
    
if __name__ == '__main__':
    pl.seed_everything(0)

    parser = exp.ArgumentParserX(base_config_path=zeus.default_config_dir / 'test.yaml', parents=[get_default_parser()])
    parser.add_argument('--ckpt', type=str, required=False, help='Path to ckpt file.')
    parser.add_argument('--weight', type=str, required=False, default='default',
                        help="Overwrite the weight defined by --ckpt. "
                             "Explicitly set to 'none' so that no weight will be loaded.")
    parser.add_argument('--nosync', action='store_true', help='Do not synchronize nas even if forced.')
    parser.add_argument('--record', nargs='*',
                        help='Whether or not to store evaluation data. add name to specify save path.')
    parser.add_argument('--focus', type=str, default="none", help='Sample to focus')

    known_args = parser.parse_known_args()[0]
    args_ckpt = None

    if args_ckpt is not None:
        if args_ckpt.startswith("wdb:"):
            wdb_run, args_ckpt = wdb.get_wandb_run(args_ckpt, wdb_base=zeus.config.wandb.base, default_ckpt="last")
            assert args_ckpt is not None, "Please specify checkpoint version!"
            assert args_ckpt.exists(), "Selected checkpoint does not exist!"
            model_args = omegaconf.OmegaConf.create(wdb.recover_from_wandb_config(wdb_run.config))
        else:
            model_yaml_path = Path(known_args.ckpt).parent.parent / "hparams.yaml"
            model_args = exp.parse_config_yaml(model_yaml_path)
    else:
        model_args = None
    args = parser.parse_args(additional_args=model_args)

    if args.nosync:
        # Force not to sync to shorten bootstrap time.
        os.environ['NO_SYNC'] = '1'

    if args.gpus is None:
        args.gpus = 1

    trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**args), logger=None, max_epochs=1, inference_mode=False)
    net_module = importlib.import_module("models." + args.model).Model

    # --ckpt & --weight logic:
    if args.weight == 'default':
        ckpt_path = args_ckpt
    elif args.weight == 'none':
        ckpt_path = None
    else:
        ckpt_path = args.weight

    ckpt_path = known_args.ckpt
    try:
        if ckpt_path is not None:
            net_model = net_module.load_from_checkpoint(ckpt_path, hparams=args)
        else:
            net_model = net_module(args)
        net_model.overfit_logger = zeus.OverfitLoggerNull()
        
        Trainer_test = False
        if Trainer_test:
            with exp.pt_profile_named("trainer.test", "test.json"):
                test_result = trainer.test(net_model)

            # Usually, PL will output aggregated test metric from LoggerConnector (obtained from trainer.results)
            #   However, as we patch self.log for test. We would print that ourselves.
            net_model.print_test_logs()

        else: 
            """ test from reconstructor """
            intake_id = 148
            # Initialize the ScanNetDataset
            dataset = ScanNetDataset(split='val', partial_input=True, base_path='/localhome/zla247/theia1_data/scannetv2', over_fitting=True, intake_start=intake_id,  num_input_points=10000, std_dev=0.00, uniform_sampling=True)
            # dataset = SceneNNDataset(split='val', partial_input=True, dataset_folder='/localhome/zla247/theia2_data/scenenn_seg_76_raw/scenenn_sub_data', over_fitting=True, num_input_point=10000, std_dev=0.00)
            # dataset = SyntheticRoomDataset(split='val', partial_input=True, base_path='/localhome/zla247/theia2_data/synthetic_data/synthetic_room_dataset', over_fitting=True, intake_start=intake_id, num_input_points=10000, std_dev=0.00)
            # dataset = CarlaDataset(split='val', partial_input=True, over_fitting=True, intake_start=2, num_input_points=10000, std_dev=0.00)

            # Initialize a device
            device = torch.device("cuda")
            net_model.network.to(device).eval().requires_grad_(False)
            # Prepare to accumulate evaluation metrics
            accumulated_eval_dict = {metric: 0.0 for metric in UnitMeshEvaluator.ALL_METRICS}
            total_scenes = len(dataset)
            # Start the timer
            start_time = time.time()
            total_reconstruction_duration = 0.0
            total_forward_duration = 0.0
            total_encoder_duration = 0.0
            total_solver_duration = 0.0
            total_evaluate_duration = 0.0
            total_grid_duration = 0.0
            total_dmc_duration = 0.0
            total_sdf_error = 0.0
            total_normal_error = 0.0
            results_dict = []
            for data_id in tqdm(range(total_scenes), desc="Processing scenes"):
                # Get the data for the current scene
                data = dataset._get_item(data_id, np.random.default_rng())
                # Move data to the desired device and add noise if necessary
                sparse_input_xyz = torch.from_numpy(data['partial_input']).float().to(device)
                sparse_input_normal = torch.from_numpy(data['partial_normal']).float().to(device)
                # Reconstruct the scene
                process_start = time.time()
                forward_start = time.time()
                reconstructor = nksr.Reconstructor(net_model.network, device)
                field, encoder_time, solver_time  = reconstructor.reconstruct(sparse_input_xyz, sparse_input_normal, voxel_size=0.02, solver_max_iter= 2000)
                forward_end = time.time()
                total_encoder_duration += encoder_time
                total_solver_duration += solver_time
                mesh_res, dmc_time, evaluate_time, grid_time = field.extract_dual_mesh(mise_iter=0, input_xyz = sparse_input_xyz, gt_xyz = data['full_input'])
                sdf_error, normal_error = field.compute_objective_function(xyz = sparse_input_xyz, gt_normals = 
                                                                          sparse_input_normal)
                total_sdf_error += sdf_error
                total_normal_error += normal_error
                dmc_time -= time.time()
                nksr_mesh = vis.mesh(mesh_res.v, mesh_res.f)
                dmc_time += time.time()
                total_evaluate_duration += evaluate_time
                total_grid_duration += grid_time
                total_dmc_duration += dmc_time
                # Calculate time taken for these three steps
                process_end = time.time()
                total_forward_duration += forward_end - forward_start
                process_duration = process_end - process_start
                total_reconstruction_duration += process_duration  # Accumulate the duration
                print(f"Time taken for the forward pass: {total_forward_duration:.2f} seconds")
                print(f"Time taken for the reconstruction process: {total_reconstruction_duration:.2f} seconds")
                print(f"Time taken for the encoder: {total_encoder_duration:.2f} seconds")
                print(f"Time taken for the solver: {total_solver_duration:.2f} seconds")
                print(f"Time taken for the DMC: {total_dmc_duration:.2f} seconds")
                print(f"Time taken for the evaluation: {total_evaluate_duration:.2f} seconds")
                print(f"Time taken for the grid: {total_grid_duration:.2f} seconds")
                print(f"Total SDF error: {total_sdf_error:.5f}")
                print(f"Total normal error: {total_normal_error:.5f}")
                # # Evaluate the reconstructed mesh
                evaluator = UnitMeshEvaluator(n_points=100000, metric_names=UnitMeshEvaluator.ESSENTIAL_METRICS)
                # evaluator = MeshEvaluator(n_points=int(5e6), metric_names=MeshEvaluator.ESSENTIAL_METRICS)
                eval_dict, translation, scale = evaluator.eval_mesh(nksr_mesh, torch.from_numpy(data['full_input']), torch.from_numpy(data['full_normal']), onet_samples=None)
                eval_dict["data_id"] = data_id
                results_dict.append(eval_dict)
                # o3d.io.write_triangle_mesh(f"../../theia2_data/Visualizations/DMC_visualizations/ScanNet-{intake_id}_NKSR.obj", nksr_mesh)

                # # Accumulate evaluation metrics
                for key in accumulated_eval_dict.keys():
                    if key in eval_dict:
                        accumulated_eval_dict[key] += eval_dict[key]
                        # Print the updated value for the current key
                        print(f"{key}: {eval_dict[key]}")
                # torch.cuda.empty_cache()
                

            # Stop the timer
            end_time = time.time()

            # Calculate the total time taken
            total_time = end_time - start_time
            print(f"Total reconstruction time for all scenes: {total_time:.2f} seconds")
            # Compute the average evaluation metrics
            average_eval_dict = {key: value / total_scenes for key, value in accumulated_eval_dict.items()}
            print("Average Evaluation Metrics:", average_eval_dict)
            # Path to the file where you want to save the results


            # Path to the file where you want to save the results
            file_path = 'results.txt'

            # Write the dictionary to the file
            with open(file_path, 'w') as file:
                for item in results_dict:
                    file.write(json.dumps(item, default=convert_non_serializable) + '\n')

            print(f'Results saved to {file_path}')

            # with exp.pt_profile_named("trainer.test", "test.json"):
            #     test_result = trainer.test(net_model)

            # # Usually, PL will output aggregated test metric from LoggerConnector (obtained from trainer.results)
            # #   However, as we patch self.log for test. We would print that ourselves.
            # net_model.print_test_logs()

    except Exception as ex:
        if isinstance(ex, bdb.BdbQuit):
            exp.logger.info("Post mortem is skipped because the exception is from Pdb. Bye!")
        elif isinstance(ex, KeyboardInterrupt):
            exp.logger.info("Keyboard Interruption. Program end normally.")
        else:
            import sys, pdb, traceback
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
            sys.exit(-1)
