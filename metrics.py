# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
import open3d as o3d
from pycg import vis, exp
from pykdtree.kdtree import KDTree


NAN_METRIC = float('nan')


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to method not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold

    # Calculate the bounds
    min_bound = np.min(pointcloud, axis=0)
    max_bound = np.max(pointcloud, axis=0)

    # Calculate the scale and translate factors
    scale = 1.0 / (max_bound - min_bound).max()
    translate = -0.5 * (max_bound + min_bound)

    # Normalize the pointcloud to the unit cube centered at the origin
    pointcloud_normalized = (pointcloud + translate) * scale
    return pointcloud_normalized, scale, translate

def normalize_pointcloud_to_unit_cube(pointcloud):
    # Calculate the bounds
    min_bound = np.min(pointcloud, axis=0)
    max_bound = np.max(pointcloud, axis=0)

    # Calculate the scale and translate factors
    scale = 1.0 / (max_bound - min_bound).max()
    translate = -0.5 * (max_bound + min_bound)

    # Normalize the pointcloud to the unit cube centered at the origin
    pointcloud_normalized = (pointcloud + translate) * scale
    return pointcloud_normalized, scale, translate

def sample_and_normalize_pointclouds(pointcloud, pointcloud_tgt, num_samples=100000):
    num_dense_samples = 200000
    num_gt_samples = 100000
    indices = np.random.choice(pointcloud.shape[0], num_dense_samples, replace=True)
    sampled_pointcloud = pointcloud[indices, :]

    indices_tgt = np.random.choice(pointcloud_tgt.shape[0], num_gt_samples, replace=True)
    sampled_pointcloud_tgt = pointcloud_tgt[indices_tgt, :]

    min_range = np.min(pointcloud_tgt, axis=0)
    max_range = np.max(pointcloud_tgt, axis=0)

    range_expansion = 0.05
    min_range -= range_expansion
    max_range += range_expansion

    in_range_indices = np.all(np.logical_and(pointcloud >= min_range, pointcloud <= max_range), axis=1)
    filtered_pointcloud = pointcloud[in_range_indices]

    # Step 5: Randomly sample 100000 points from the filtered pointcloud
    if filtered_pointcloud.shape[0] < num_samples:
        raise ValueError("Not enough points in the filtered pointcloud to sample the desired number of points.")
    
    indices = np.random.choice(pointcloud.shape[0], num_samples, replace=False)
    sampled_pointcloud = pointcloud[indices, :]

    sampled_pointcloud_tgt, scale, translate = normalize_pointcloud_to_unit_cube(sampled_pointcloud_tgt)
    # sampled_pointcloud, _, _= normalize_pointcloud_to_unit_cube(sampled_pointcloud)
    sampled_pointcloud =  (sampled_pointcloud + translate) * scale

    return sampled_pointcloud, sampled_pointcloud_tgt, translate, scale

class MeshEvaluator:

    ESSENTIAL_METRICS = [
        'chamfer-L1', 'f-score', 'normals'
    ]
    ALL_METRICS = [
        'completeness', 'accuracy', 'normals completeness', 'normals accuracy', 'normals',
        'completeness2', 'accuracy2', 'chamfer-L2',
        'chamfer-L1', 'f-precision', 'f-recall', 'f-score', 'f-score-15', 'f-score-20'
    ]

    """
    Mesh evaluation class that handles the mesh evaluation process. Returned dict has meaning:
        - completeness:             mean distance from all gt to pd.
        - accuracy:                 mean distance from all pd to gt.
        - chamfer-l1/l2:            average of the above two. [Chamfer distance]
        - f-score(/-15/-20):        [F-score], computed at the threshold of 0.01, 0.015, 0.02.
        - normals completeness:     mean normal alignment (0-1) from all gt to pd.
        - normals accuracy:         mean normal alignment (0-1) from all pd to gt.
        - normals:                  average of the above two, i.e., [Normal Consistency Score] (0-1)
    Args:
        n_points (int): number of points to be used for evaluation
    """

    def __init__(self, n_points=100000, metric_names=ALL_METRICS):
        self.n_points = n_points
        self.thresholds = np.array([0.01, 0.015, 0.02, 0.002, 0.1])
        self.fidx = [0, 1, 2, 3, 4]
        self.metric_names = metric_names

    def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt, onet_samples=None):
        """
        Evaluates a mesh.
        :param mesh: (o3d.geometry.TriangleMesh) mesh which should be evaluated
        :param pointcloud_tgt: np (Nx3) ground-truth xyz
        :param normals_tgt: np (Nx3) ground-truth normals
        :param onet_samples: (Nx3, N) onet samples and occupancy (latter is 1 inside, 0 outside)
        :return: metric-dict
        """
        if isinstance(pointcloud_tgt, torch.Tensor):
            pointcloud_tgt = pointcloud_tgt.detach().cpu().numpy().astype(float)

        if isinstance(normals_tgt, torch.Tensor):
            normals_tgt = normals_tgt.detach().cpu().numpy().astype(float)

        # Triangle normal is used to be consistent with SAP.
        try:
            # Ensure same random seed for reproducibility
            o3d.utility.random.seed(0)
            sampled_pcd = mesh.sample_points_uniformly(
                number_of_points=self.n_points, use_triangle_normal=True)
            pointcloud = np.asarray(sampled_pcd.points)
            normals = np.asarray(sampled_pcd.normals)
        except RuntimeError:    # Sample error.
            pointcloud = np.zeros((0, 3))
            normals = np.zeros((0, 3))

        out_dict = self._evaluate(
            pointcloud, pointcloud_tgt, normals, normals_tgt, onet_samples, mesh)

        return out_dict

    def _evaluate(self, pointcloud, pointcloud_tgt, normals=None, normals_tgt=None, onet_samples=None, mesh=None):
        """
        Evaluates a point cloud.
        :param pointcloud: np (Mx3) predicted xyz
        :param pointcloud_tgt:  np (Nx3) ground-truth xyz
        :param normals: np (Mx3) predicted normals
        :param normals_tgt: np (Nx3) ground-truth normals
        :return: metric-dict
        """
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            exp.logger.warning('Empty pointcloud / mesh detected! Return NaN metric!')
            return {k: NAN_METRIC for k in self.metric_names}

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, self.thresholds)
        completeness2 = completeness ** 2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        precision = get_threshold_percentage(accuracy, self.thresholds)
        accuracy2 = accuracy ** 2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamfer_l2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
                0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamfer_l1 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            for i in range(len(precision))
        ]

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamfer_l2,
            'chamfer-L1': chamfer_l1,
            'f-precision': precision[self.fidx[0]],
            'f-recall': recall[self.fidx[0]],
            'f-score': F[self.fidx[0]],  # threshold = 1.0%
            'f-score-15': F[self.fidx[1]],  # threshold = 1.5%
            'f-score-20': F[self.fidx[2]],  # threshold = 2.0%
            # -- F-outdoor
            'f-precision-outdoor': precision[self.fidx[4]],
            'f-recall-outdoor': recall[self.fidx[4]],
            'f-score-outdoor': F[self.fidx[4]]
        }

        if onet_samples is not None:
            if len(mesh.triangles) == 0:
                out_dict['o3d-iou'] = NAN_METRIC
            else:
                onet_pd_occ = vis.RayDistanceQuery(mesh).compute_occupancy(onet_samples[0])
                onet_gt_occ = onet_samples[1]
                iou = np.sum(np.logical_and(onet_pd_occ, onet_gt_occ)) / \
                      (np.sum(np.logical_or(onet_pd_occ, onet_gt_occ)) + 1.0e-6)
                out_dict['o3d-iou'] = iou

        return {
            k: out_dict[k] for k in self.metric_names
        }

class UnitMeshEvaluator:

    ESSENTIAL_METRICS = [
        'chamfer-L1', 'f-score', 'normals'
    ]
    ALL_METRICS = [
        'completeness', 'accuracy', 'normals completeness', 'normals accuracy', 'normals',
        'completeness2', 'accuracy2', 'chamfer-L2',
        'chamfer-L1', 'f-precision', 'f-recall', 'f-score', 'f-score-15', 'f-score-20'
    ]

    """
    Mesh evaluation class that handles the mesh evaluation process. Returned dict has meaning:
        - completeness:             mean distance from all gt to pd.
        - accuracy:                 mean distance from all pd to gt.
        - chamfer-l1/l2:            average of the above two. [Chamfer distance]
        - f-score(/-15/-20):        [F-score], computed at the threshold of 0.01, 0.015, 0.02.
        - normals completeness:     mean normal alignment (0-1) from all gt to pd.
        - normals accuracy:         mean normal alignment (0-1) from all pd to gt.
        - normals:                  average of the above two, i.e., [Normal Consistency Score] (0-1)
    Args:
        n_points (int): number of points to be used for evaluation
    """

    def __init__(self, n_points=100000, metric_names=ALL_METRICS):
        self.n_points = n_points
        self.thresholds = np.array([0.01, 0.015, 0.02, 0.002, 0.1])
        self.fidx = [0, 1, 2, 3, 4]
        self.metric_names = metric_names

    def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt, onet_samples=None):
        """
        Evaluates a mesh.
        :param mesh: (o3d.geometry.TriangleMesh) mesh which should be evaluated
        :param pointcloud_tgt: np (Nx3) ground-truth xyz
        :param normals_tgt: np (Nx3) ground-truth normals
        :param onet_samples: (Nx3, N) onet samples and occupancy (latter is 1 inside, 0 outside)
        :return: metric-dict
        """
        if isinstance(pointcloud_tgt, torch.Tensor):
            pointcloud_tgt = pointcloud_tgt.detach().cpu().numpy().astype(float)

        if isinstance(normals_tgt, torch.Tensor):
            normals_tgt = normals_tgt.detach().cpu().numpy().astype(float)

        # Triangle normal is used to be consistent with SAP.
        try:
            # Ensure same random seed for reproducibility
            o3d.utility.random.seed(0)
            sampled_pcd = mesh.sample_points_uniformly(
                number_of_points=200000, use_triangle_normal=True)
            pointcloud = np.asarray(sampled_pcd.points)
            normals = np.asarray(sampled_pcd.normals)
        except RuntimeError:    # Sample error.
            pointcloud = np.zeros((0, 3))
            normals = np.zeros((0, 3))

        # out_dict = self._evaluate(
            # pointcloud, pointcloud_tgt, normals, normals_tgt, onet_samples, mesh)
        out_dict = self._evaluate(
            pointcloud, pointcloud_tgt, None, None, None)
        return out_dict

    def _evaluate(self, pointcloud, pointcloud_tgt, normals=None, normals_tgt=None, onet_samples=None, mesh=None):
        """
        Evaluates a point cloud.
        :param pointcloud: np (Mx3) predicted xyz
        :param pointcloud_tgt:  np (Nx3) ground-truth xyz
        :param normals: np (Mx3) predicted normals
        :param normals_tgt: np (Nx3) ground-truth normals
        :return: metric-dict
        """
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            exp.logger.warning('Empty pointcloud / mesh detected! Return NaN metric!')
            return {k: NAN_METRIC for k in self.metric_names}

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        pointcloud, pointcloud_tgt, translation, scale = sample_and_normalize_pointclouds(pointcloud, pointcloud_tgt)

        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, self.thresholds)
        completeness2 = completeness ** 2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        precision = get_threshold_percentage(accuracy, self.thresholds)
        accuracy2 = accuracy ** 2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamfer_l2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
                0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamfer_l1 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            for i in range(len(precision))
        ]

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamfer_l2,
            'chamfer-L1': chamfer_l1,
            'f-precision': precision[self.fidx[0]],
            'f-recall': recall[self.fidx[0]],
            'f-score': F[self.fidx[0]],  # threshold = 1.0%
            'f-score-15': F[self.fidx[1]],  # threshold = 1.5%
            'f-score-20': F[self.fidx[2]],  # threshold = 2.0%
            # -- F-outdoor
            'f-precision-outdoor': precision[self.fidx[4]],
            'f-recall-outdoor': recall[self.fidx[4]],
            'f-score-outdoor': F[self.fidx[4]]
        }

        if onet_samples is not None:
            if len(mesh.triangles) == 0:
                out_dict['o3d-iou'] = NAN_METRIC
            else:
                onet_pd_occ = vis.RayDistanceQuery(mesh).compute_occupancy(onet_samples[0])
                onet_gt_occ = onet_samples[1]
                iou = np.sum(np.logical_and(onet_pd_occ, onet_gt_occ)) / \
                      (np.sum(np.logical_or(onet_pd_occ, onet_gt_occ)) + 1.0e-6)
                out_dict['o3d-iou'] = iou

        # return {
        #     k: out_dict[k] for k in self.metric_names
        # }
        return out_dict, translation, scale