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
import torch
import open3d as o3d
from torch.nn import functional as F

from pycg import vis, exp
from pathlib import Path
import numpy as np
from metrics import UnitMeshEvaluator

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

        reconstructor = nksr.Reconstructor(net_model.network, device)
        field = reconstructor.reconstruct(sparse_input_xyz, sparse_input_normal, voxel_size=0.02)
        mesh_res = field.extract_dual_mesh(mise_iter=0)
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
        print(eval_dict)

        o3d.io.write_triangle_mesh("../../projects/data/Visualizations/0nksr_mesh_voxel_0.02.obj", nksr_mesh)

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
