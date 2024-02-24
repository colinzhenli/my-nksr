from pathlib import Path

import numpy as np
import torch
import os

from dataset.base import DatasetSpec as DS
from dataset.base import RandomSafeDataset
from dataset.transforms import ComposedTransforms


class ScanNetDataset(RandomSafeDataset):
    def __init__(self, spec, split, transforms=None, partial_input=False,
                 random_seed=0, hparams=None, skip_on_error=False, custom_name="scannet", custom_scenes=None,
                 **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)
        self.skip_on_error = skip_on_error
        self.custom_name = custom_name
        self.over_fitting = kwargs.get("over_fitting", False)
        self.intake_start = kwargs.get("intake_start", 0)
        self.take = kwargs.get("take", 4)
        self.num_input_points = kwargs.get("num_input_points", 5000)
        self.std_dev = kwargs.get("std_dev", 0.00)

        assert DS.GT_MESH not in spec and DS.GT_MESH_SOUP not in spec
        self.split = 'train' if self.over_fitting else split # use only train set for overfitting
        self.split = 'val'
        self.spec = self.sanitize_specs(
            spec, [DS.SCENE_NAME, DS.INPUT_PC, DS.TARGET_NORMAL, DS.GT_DENSE_PC, DS.GT_DENSE_NORMAL])
        self.transforms = ComposedTransforms(transforms)
        self.base_path = Path(kwargs.get("base_path", None))

        if self.split == "test":
            with (self.base_path / "metadata" / "scannetv2_test.txt").open() as f:
                self.scenes = [t.strip() for t in f.readlines()]
        elif self.split == "custom":
            assert custom_scenes is not None
            self.scenes = custom_scenes
        elif self.split == "train":
            with (self.base_path / "metadata" / "scannetv2_train.txt").open() as f:
                self.scenes = [t.strip() for t in f.readlines()]
        else:
            with (self.base_path / "metadata" / "scannetv2_val.txt").open() as f:
                self.scenes = [t.strip() for t in f.readlines()]
        
        if self.over_fitting:
            self.scenes = self.scenes[self.intake_start:self.take+self.intake_start]

        self.scenes = ['scene0221_00']
        self.hparams = hparams
        self.partial_input = partial_input

    def __len__(self):
        return len(self.scenes)

    def get_name(self):
        return f"{self.custom_name}-{self.split}"

    def get_short_name(self):
        return f"{self.custom_name}"

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
