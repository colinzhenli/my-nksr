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
import torch
import open3d as o3d
from torch.nn import functional as F

from pycg import vis, exp
from pathlib import Path
import numpy as np
from metrics import UnitMeshEvaluator
from torch.utils.data import Dataset
from tqdm import tqdm


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
            self.split = 'val'
            self.scenes = ['scene0221_00']
        
    def __len__(self):
        return len(self.scenes)

    def _get_item(self, data_id, rng):
        scene_name = self.scenes[data_id]

        data = {}
        scene_path = os.path.join(self.base_path, self.split, f"{scene_name}.pth")
        full_data = torch.load(scene_path)
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

        """ test from reconstructor """
        # Initialize the ScanNetDataset
        dataset = ScanNetDataset(split='val', partial_input=True, base_path='/localhome/zla247/data/scannetv2', over_fitting=False, num_input_points=10000, std_dev=0.00)
        # Initialize a device
        device = torch.device("cpu")
        # Prepare to accumulate evaluation metrics
        accumulated_eval_dict = {metric: 0.0 for metric in UnitMeshEvaluator.ALL_METRICS}
        total_scenes = len(dataset)
        # Start the timer
        start_time = time.time()
        for data_id in tqdm(range(total_scenes), desc="Processing scenes"):
            # Get the data for the current scene
            data = dataset._get_item(data_id, np.random.default_rng())
            # Move data to the desired device and add noise if necessary
            sparse_input_xyz = torch.from_numpy(data['partial_input']).float().to(device)
            sparse_input_normal = torch.from_numpy(data['partial_normal']).float().to(device)
            # Reconstruct the scene
            reconstructor = nksr.Reconstructor(net_model.network, device)
            field = reconstructor.reconstruct(sparse_input_xyz, sparse_input_normal, voxel_size=0.02)
            mesh_res = field.extract_dual_mesh(mise_iter=0)
            nksr_mesh = vis.mesh(mesh_res.v, mesh_res.f)
            # Evaluate the reconstructed mesh
            evaluator = UnitMeshEvaluator(n_points=100000, metric_names=UnitMeshEvaluator.ESSENTIAL_METRICS)
            eval_dict, translation, scale = evaluator.eval_mesh(nksr_mesh, torch.from_numpy(data['full_input']), torch.from_numpy(data['full_normal']), onet_samples=None)
            # o3d.io.write_triangle_mesh("../../projects/data/Visualizations/Epoch13_ResNet_Attention-no-growing_0.02.obj", nksr_mesh)
            # Accumulate evaluation metrics
            for key in accumulated_eval_dict.keys():
                if key in eval_dict:
                    accumulated_eval_dict[key] += eval_dict[key]
                    # Print the updated value for the current key
                    print(f"{key}: {eval_dict[key]}")

        # Stop the timer
        end_time = time.time()

        # Calculate the total time taken
        total_time = end_time - start_time
        print(f"Total reconstruction time for all scenes: {total_time:.2f} seconds")
        # Compute the average evaluation metrics
        average_eval_dict = {key: value / total_scenes for key, value in accumulated_eval_dict.items()}
        print("Average Evaluation Metrics:", average_eval_dict)

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
