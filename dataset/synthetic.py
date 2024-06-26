from pathlib import Path

import numpy as np
import torch
import os

from dataset.base import DatasetSpec as DS
from dataset.base import RandomSafeDataset
from dataset.transforms import ComposedTransforms


class SyntheticRoomDataset(RandomSafeDataset):
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


        # Flip the y and z axes for points and normals and move to positive quadrant
        points = points[:, [0, 2, 1]]
        normals = normals[:, [0, 2, 1]]
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
