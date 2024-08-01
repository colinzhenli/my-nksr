from pathlib import Path

import numpy as np
import torch
import os

from dataset.base import DatasetSpec as DS
from dataset.base import RandomSafeDataset
from dataset.transforms import ComposedTransforms
from plyfile import PlyData



class SceneNNDataset(RandomSafeDataset):
    def __init__(self, spec, split, transforms=None, partial_input=False,
                 random_seed=0, hparams=None, skip_on_error=False, custom_name="synthetic", custom_scenes=None,
                 **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)
        self.scale = 2.2 # Emperical scale to transfer back to physical scale
        self.skip_on_error = skip_on_error
        self.custom_name = custom_name
        self.dataset_folder = kwargs.get("base_path", None)
        self.file_name = 'pointcloud'
        self.multi_files = kwargs.get("multi_files", None)
        categories = kwargs.get("classes", None)
        self.over_fitting = kwargs.get("over_fitting", False)
        self.intake_start = kwargs.get("intake_start", 0)
        self.take = kwargs.get("take", 1)
        self.num_input_points = kwargs.get("num_input_points", 5000)
        self.std_dev = kwargs.get("std_dev", 0.00)
        self.std_dev *= 2

        assert DS.GT_MESH not in spec and DS.GT_MESH_SOUP not in spec
        self.split = 'val' if self.over_fitting else split # use only train set for overfitting
        # self.split = 'val'
        self.spec = self.sanitize_specs(
            spec, [DS.SCENE_NAME, DS.INPUT_PC, DS.TARGET_NORMAL, DS.GT_DENSE_PC, DS.GT_DENSE_NORMAL])
        self.hparams = hparams
        self.partial_input = partial_input
        self.transforms = ComposedTransforms(transforms)

        self.filenames = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(self.dataset_folder) for f in filenames if f.endswith('.ply')])
        if self.over_fitting:
            self.filenames = self.filenames[self.intake_start:self.take+self.intake_start]

    def __len__(self):
        return len(self.filenames)

    def _get_item(self, data_id, rng):
        data = {}
        scene_filename = self.filenames[data_id]

        ply_data = PlyData.read(scene_filename)
        vertex = ply_data['vertex']
        pos = np.stack([vertex[t] for t in ('x', 'y', 'z')], axis=1)
        nls = np.stack([vertex[t] for t in ('nx', 'ny', 'nz')], axis=1) if 'nx' in vertex and 'ny' in vertex and 'nz' in vertex else np.zeros_like(pos)

        if len(pos) > 200000:
            indices = np.random.choice(len(pos), 200000, replace=False)
            pos = pos[indices]
            nls = nls[indices]
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

        if DS.SCENE_NAME in self.spec:
            data[DS.SCENE_NAME] = scene_name

        if DS.GT_DENSE_PC in self.spec:
            data[DS.GT_DENSE_PC] = full_points

        if DS.GT_DENSE_NORMAL in self.spec:
            data[DS.GT_DENSE_NORMAL] = full_normals

        if DS.INPUT_PC in self.spec:
            data[DS.INPUT_PC] = partial_points

        if DS.TARGET_NORMAL in self.spec:
            data[DS.TARGET_NORMAL] = partial_normals

        if self.transforms is not None:
            data = self.transforms(data, rng)

        return data
