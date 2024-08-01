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
        self.split = 'val' if self.over_fitting else split # use only train set for overfitting
        # self.split = 'val'
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
            self.scenes = ['scene0221_00']
            # self.scenes = self.scenes[self.intake_start:self.take+self.intake_start]

        # self.scenes = ['scene0221_00']
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
        self.uniform_sampling = False

        if self.num_input_points != -1:
            if not self.uniform_sampling:
                # Number of blocks along each axis
                num_blocks = 3
                total_blocks = num_blocks ** 3
                self.common_difference = 25
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
