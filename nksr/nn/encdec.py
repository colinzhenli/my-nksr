# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


# Code adapted from Convolutional Occupancy Networks.
#   As some arguments are found to be less improving, we removed them.

import torch
from torch import Tensor, nn
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
from torch_scatter import scatter_mean, scatter_max
from nksr.svh import SparseFeatureHierarchy
# from knn_cuda import KNN
from sklearn.neighbors import NearestNeighbors


class ResnetBlockFC(nn.Module):
    """ Fully connected ResNet Block class. """
    def __init__(self, size_in: int, size_out: int = None, size_h: int = None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name="",
            instance_norm=False,
            instance_norm_func=None
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)

class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size, eps=1e-6, momentum=0.99))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)

class _BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)

class ActivationConv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=_BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm1d
        )

    def forward(self, x):
        # Reshape input from [B, K, C] to [B, C, K]
        x = x.transpose(1, 2)
        # Apply convolution and other operations defined in __init__
        x = super().forward(x)
        # Reshape output back to [B, K, C]
        x = x.transpose(1, 2)
        return x
    
class PointEncoder(nn.Module):
    """ PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    """

    def __init__(self,
                 dim: int,
                 c_dim: int = 32,
                 hidden_dim: int = 32,
                 n_blocks: int = 3):
        super().__init__()

        self.c_dim = c_dim
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim)
            for _ in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.hidden_dim = hidden_dim

    def forward(self,
                pts_xyz: torch.Tensor,
                pts_feature: torch.Tensor,
                svh: SparseFeatureHierarchy,
                depth: int = 0):

        grid = svh.grids[depth]
        assert grid is not None, "Grid structure is not built for PointEncoder!"

        # Get voxel idx
        pts_xyz = grid.world_to_grid(pts_xyz)
        vid = grid.ijk_to_index(pts_xyz.round().int())

        # Map coordinates to local voxel
        pts_xyz = (pts_xyz + 0.5) % 1
        pts_mask = vid != -1
        vid, pts_xyz = vid[pts_mask], pts_xyz[pts_mask]

        # Feature extraction
        if pts_feature is None:
            pts_feature = self.fc_pos(pts_xyz)
        else:
            pts_feature = pts_feature[pts_mask]
            pts_feature = self.fc_pos(torch.cat([pts_xyz, pts_feature], dim=1))
        pts_feature = self.blocks[0](pts_feature)
        for block in self.blocks[1:]:
            pooled = scatter_max(pts_feature, vid, dim=0, dim_size=grid.num_voxels)[0]
            pooled = pooled[vid]
            pts_feature = torch.cat([pts_feature, pooled], dim=1)
            pts_feature = block(pts_feature)

        c = self.fc_c(pts_feature)
        c = scatter_mean(c, vid, dim=0, dim_size=grid.num_voxels)
        return c


class MultiscalePointDecoder(nn.Module):
    def __init__(self,
                 c_each_dim: int = 16,
                 multiscale_depths: int = 4,
                 p_dim: int = 3,
                 out_dim: int = 1,
                 hidden_size: int = 32,
                 n_blocks: int = 2,
                 aggregation: str = 'cat',
                 out_init: float = None,
                 coords_depths: list = None):

        if aggregation == 'cat':
            c_dim = c_each_dim * multiscale_depths
        elif aggregation == 'sum':
            c_dim = c_each_dim
        else:
            raise NotImplementedError

        if coords_depths is None:
            coords_depths = list(range(multiscale_depths))
        coords_depths = sorted(coords_depths)

        super().__init__()
        self.c_dim = c_dim
        self.c_each_dim = c_each_dim
        self.n_blocks = n_blocks
        self.multiscale_depths = multiscale_depths
        self.aggregation = aggregation
        self.coords_depths = coords_depths

        self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)])
        self.fc_p = nn.Linear(p_dim * len(coords_depths), hidden_size)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for _ in range(n_blocks)
        ])
        self.fc_out = nn.Linear(hidden_size, out_dim)
        self.out_dim = out_dim

        # Init parameters
        if out_init is not None:
            nn.init.zeros_(self.fc_out.weight)
            nn.init.constant_(self.fc_out.bias, out_init)

    def forward(self,
                xyz: torch.Tensor,
                svh: SparseFeatureHierarchy,
                multiscale_feat: dict):

        p_feats = []
        for did in self.coords_depths:
            vs = svh.grids[did].voxel_size
            p = (xyz % vs) / vs - 0.5
            p_feats.append(p)
        p = torch.cat(p_feats, dim=1)

        c_feats = []
        for did in range(self.multiscale_depths):
            if svh.grids[did] is None:
                c = torch.zeros((xyz.size(0), self.c_each_dim), device=xyz.device)
            else:
                c = svh.grids[did].sample_trilinear(xyz, multiscale_feat[did])
            c_feats.append(c)

        if self.aggregation == 'cat':
            c = torch.cat(c_feats, dim=1)
        else:
            c = sum(c_feats)

        net = self.fc_p(p)
        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)
            net = self.blocks[i](net)
        out = self.fc_out(F.relu(net))

        return out

class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out, alpha, neighbor_level_mlp, activation_fn):
        super().__init__()
        self.alpha = alpha
        if neighbor_level_mlp:
            self.fc_layers = nn.Sequential(
                nn.Linear(d_in, d_in),  
                activation_fn,              
                nn.Linear(d_in, d_in)   
            )
        else:
            self.fc_layers = nn.Linear(d_in, d_in)

        self.mlp = ActivationConv1d(d_in, d_out, kernel_size=1,bn=True, activation=activation_fn)

    def forward(self, feature_set, alpha, mask=None):
        if mask is not None:
            mask = mask==False
            # Flatten and filter feature_set according to mask
            N, K, C = feature_set.shape
            flat_features = feature_set.reshape(-1, C)  # Shape (N*K, C)
            valid_indices = mask.flatten().nonzero().squeeze(-1)
            filtered_features = flat_features[valid_indices]  # Shape (M, C), M << N*K
            # Process filtered features with fc layers
            fc_output = self.fc_layers(filtered_features)  # Shape (M, C)
            # Map fc layer outputs back to original shape according to mask
            # Create an output tensor filled with zeros
            output = torch.zeros_like(flat_features)
            # Place fc_output back according to valid_indices
            output[valid_indices] = fc_output

            # Reshape back to (N, K, C) and process with mlp if necessary
            output = output.reshape(N, K, C)
        att_activation = self.fc_layers(feature_set)
        att_scores = F.softmax(att_activation, dim=1) # M, K, hidden_dim + feature_dim
        if self.alpha:
            att_scores = alpha * att_scores
            att_scores = att_scores / (torch.sum(att_scores, dim=1, keepdim=True) + 1e-5)
        # normalize att_scores #
        f_agg = feature_set * att_scores #M, K, hidden_dim + feature_dim
        f_agg = torch.sum(f_agg, dim=1, keepdim=True) # M, 1, hidden_dim + feature_dim
        f_agg = self.mlp(f_agg) #M, 1, hidden_dim + feature_dim
        return f_agg #M, 1, hidden_dim

class CoordsEncoder(nn.Module):
    def __init__(
        self,
        input_dims: int = 3,
        include_input: bool = True,
        max_freq_log2: int = 9,
        num_freqs: int = 10,
        log_sampling: bool = True,
        periodic_fns: Tuple[Callable, Callable] = (torch.sin, torch.cos)
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_embedding_fn()

    def create_embedding_fn(self) -> None:
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs: Tensor) -> Tensor:
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

""" Attention from RangeUDF """
class AttentionMultiscalePointDecoder(nn.Module):
    def __init__(self,
                 c_each_dim: int = 16,
                 multiscale_depths: int = 4,
                 p_dim: int = 3,
                 out_dim: int = 1,
                 hidden_size: int = 32,
                 n_blocks: int = 2,
                 aggregation: str = 'cat',
                 out_init: float = None,
                 coords_depths: list = None,
                 alpha: bool = False,
                 knn_mask: bool = False,
                 neighbor_level_mlp: bool = True):
        # aggregation = 'sum'
        if aggregation == 'cat':
            c_dim = c_each_dim * multiscale_depths
        elif aggregation == 'sum':
            c_dim = c_each_dim
        else:
            raise NotImplementedError

        if coords_depths is None:
            coords_depths = list(range(multiscale_depths))
        coords_depths = sorted(coords_depths)

        super().__init__()
        self.k_neighbors = 8
        # self.coords_enc = CoordsEncoder(p_dim)  
        # self.enc_dim = self.coords_enc.out_dim 
        self.enc_dim = hidden_size
        self.coords_enc = nn.Linear(p_dim, hidden_size)
        self.c_dim = c_dim 
        self.c_each_dim = c_each_dim
        # self.n_blocks = n_blocks
        self.num_hidden_layers_after = int(n_blocks/2)
        self.num_hidden_layers_before = int(n_blocks/2)
        self.multiscale_depths = multiscale_depths
        self.aggregation = aggregation
        self.coords_depths = coords_depths
        self.alpha = alpha
        self.knn_mask = knn_mask
        self.neighbor_level_mlp = neighbor_level_mlp
        self.activation_fn = self.get_activation(activation_str='LeakyReLU', negative_slope=0.01)
    
        if self.alpha: #alpha feature map
            self.alpha_map = nn.Linear(c_each_dim, 1)
            self.sigmoid = nn.Sigmoid()

        self.att_pooling_layers = nn.ModuleList({
            Att_pooling(self.enc_dim + self.c_each_dim, self.enc_dim + self.c_each_dim, self.alpha, self.neighbor_level_mlp, self.activation_fn)
            for d in range(self.multiscale_depths)  # Assuming you need a layer for each scale depth
        })

        if aggregation == 'cat':
            self.in_layer = nn.Sequential(nn.Linear(4* (self.c_each_dim + self.enc_dim), hidden_size), self.activation_fn)
            self.skip_proj = nn.Sequential(nn.Linear(4* (self.c_each_dim + self.enc_dim),hidden_size), self.activation_fn)
        else:
            self.in_layer = nn.Sequential(nn.Linear(self.c_each_dim + self.enc_dim, hidden_size), self.activation_fn)
            self.skip_proj = nn.Sequential(nn.Linear(self.c_each_dim + self.enc_dim,hidden_size), self.activation_fn)

        if self.neighbor_level_mlp:
            self.fc_out = nn.Linear(hidden_size, out_dim)
        else:
            before_skip = []
            for _ in range(self.num_hidden_layers_before):
                before_skip.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), self.activation_fn))
            self.before_skip = nn.Sequential(*before_skip)

            after_skip = []
            for _ in range(self.num_hidden_layers_after):
                after_skip.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), self.activation_fn))
            after_skip.append(nn.Linear(hidden_size, out_dim))
            # if self.supervision == 'UDF':
            #     after_skip.append(self.activation_fn)
            # else:
            after_skip.append(nn.Tanh())
            self.after_skip = nn.Sequential(*after_skip)

        # self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)])
        # self.fc_p = nn.Linear(p_dim * len(coords_depths), hidden_size)
        # self.blocks = nn.ModuleList([
        #     ResnetBlockFC(hidden_size) for _ in range(n_blocks)
        # ])
        # self.fc_out = nn.Linear(hidden_size, out_dim)
        # self.out_dim = out_dim

        # # Init parameters
        # if out_init is not None:
        #     nn.init.zeros_(self.fc_out.weight)
        #     nn.init.constant_(self.fc_out.bias, out_init)

    def get_activation(self, activation_str: str, negative_slope=0.01):
        """Return the desired activation function based on the string."""
        if activation_str == "ReLU":
            act = nn.ReLU()
        elif activation_str == "LeakyReLU":
            act = nn.LeakyReLU(negative_slope, inplace=True)
        elif activation_str == "Softplus":
            act = nn.Softplus()
        elif activation_str == "ShiftedSoftplus":
            def shifted_softplus(input_tensor):
                shifted = input_tensor - 1
                return nn.Softplus()(shifted)
            act = shifted_softplus
        else:
            raise ValueError(f"Activation {activation_str} not supported!")
        return act
    
    def forward(self,
                xyz: torch.Tensor,
                svh: SparseFeatureHierarchy,
                multiscale_feat: dict):
        # knn = KNN(k=self.k_neighbors, transpose_mode=True)
        # p_feats = []
        # for did in self.coords_depths:
        #     vs = svh.grids[did].voxel_size
        #     p = (xyz % vs) / vs - 0.5
        #     p_feats.append(p)
        # p = torch.cat(p_feats, dim=1)

        c_feats = []
        for d in range(self.multiscale_depths):
            if svh.grids[d] is None:
                interpolated_features = torch.zeros((xyz.size(0), self.c_each_dim), device=xyz.device)
            else:
                ijk_coords = svh.grids[d].active_grid_coords()
                coords = svh.grids[d].grid_to_world(ijk_coords.float()) # M, 3 voxel centers world coordinates
                # query_indices, _, _ = knn(torch.from_numpy(query_xyz).to(device), torch.from_numpy(voxel_center).to(device), 1)
                nn = NearestNeighbors(n_neighbors=self.k_neighbors)
                nn.fit(coords.detach().cpu().numpy())  # coords is an (N, 3) array
                dist, indx = nn.kneighbors(xyz.detach().cpu().numpy())  # xyz is an (M, 3) array
                indx = torch.from_numpy(indx).to(xyz.device)
                dist = torch.from_numpy(dist).to(xyz.device)

                # dist, indx = knn(coords.unsqueeze(0), xyz.unsqueeze(0))
                # dist = dist.squeeze(0)
                # indx = indx.squeeze(0)
                gathered_latents = multiscale_feat[d][indx] #N, K, C
                mask = None
                alpha = None
                if self.alpha:
                    alpha = self.sigmoid(self.alpha_map(gathered_latents)) # N, K, 1
                    if self.knn_mask:
                        mask = dist > 1.5*svh.grids[d].voxel_size
                        # alpha[mask] = 0
                        alpha = torch.where(mask.unsqueeze(-1), torch.zeros_like(alpha), alpha)
                gathered_centers = coords[indx] #N, K, 3
                gathered_query_xyz = xyz.unsqueeze(1).expand(-1, self.k_neighbors, -1) #N, K, 3
                gathered_relative_coords = gathered_query_xyz - gathered_centers #N, K, 3
                # gathered_coords = self.coords_enc.embed(gathered_relative_coords/ svh.grids[d].voxel_size)
                gathered_coords = self.coords_enc(gathered_relative_coords/ svh.grids[d].voxel_size)
                gathered_emb_and_coords = torch.cat([gathered_latents, gathered_coords], dim=-1) # M, K, C + enc_dim
                gathered_dist = torch.norm(gathered_relative_coords, dim=-1, keepdim=True) #N, K, 1
                interpolated_features = self.att_pooling_layers[d](gathered_emb_and_coords, alpha, mask) #M, 1, hidden_dim
                c_feats.append(interpolated_features.squeeze(1))

        if self.aggregation == 'cat':
            c = torch.cat(c_feats, dim=1)
        else:
            c = sum(c_feats)
        x = self.in_layer(c) # M, hidden_dim

        if self.neighbor_level_mlp:
            out = self.fc_out(F.relu(x))
            return out
        else:
            x = self.before_skip(x)
            inp_proj = self.skip_proj(c)
            x = x + inp_proj
            x = self.after_skip(x)
            return x

# """ Attention based on NKSR """
# class MultiscalePointDecoder(nn.Module):
#     def __init__(self,
#                  c_each_dim: int = 16,
#                  multiscale_depths: int = 4,
#                  p_dim: int = 3,
#                  out_dim: int = 1,
#                  hidden_size: int = 32,
#                  n_blocks: int = 2,
#                  aggregation: str = 'cat',
#                  out_init: float = None,
#                  coords_depths: list = None):

#         if aggregation == 'cat':
#             c_dim = c_each_dim * multiscale_depths
#         elif aggregation == 'sum':
#             c_dim = c_each_dim
#         else:
#             raise NotImplementedError

#         if coords_depths is None:
#             coords_depths = list(range(multiscale_depths))
#         coords_depths = sorted(coords_depths)

#         super().__init__()
#         self.k_neighbors = 8
#         self.c_dim = c_dim
#         self.c_each_dim = c_each_dim
#         self.n_blocks = n_blocks
#         self.multiscale_depths = multiscale_depths
#         self.aggregation = aggregation
#         self.coords_depths = coords_depths
#         self.activation_fn = self.get_activation(activation_str='LeakyReLU', negative_slope=0.01)
#         self.att_pooling = Att_pooling(self.c_each_dim, self.c_each_dim, self.activation_fn)

#         self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)])
#         self.fc_p = nn.Linear(p_dim * len(coords_depths), hidden_size)
#         self.blocks = nn.ModuleList([
#             ResnetBlockFC(hidden_size) for _ in range(n_blocks)
#         ])
#         self.fc_out = nn.Linear(hidden_size, out_dim)
#         self.out_dim = out_dim

#         # Init parameters
#         if out_init is not None:
#             nn.init.zeros_(self.fc_out.weight)
#             nn.init.constant_(self.fc_out.bias, out_init)

#     def get_activation(self, activation_str: str, negative_slope=0.01):
#         """Return the desired activation function based on the string."""
#         if activation_str == "ReLU":
#             act = nn.ReLU()
#         elif activation_str == "LeakyReLU":
#             act = nn.LeakyReLU(negative_slope, inplace=True)
#         elif activation_str == "Softplus":
#             act = nn.Softplus()
#         elif activation_str == "ShiftedSoftplus":
#             def shifted_softplus(input_tensor):
#                 shifted = input_tensor - 1
#                 return nn.Softplus()(shifted)
#             act = shifted_softplus
#         else:
#             raise ValueError(f"Activation {activation_str} not supported!")
#         return act
    
#     def forward(self,
#                 xyz: torch.Tensor,
#                 svh: SparseFeatureHierarchy,
#                 multiscale_feat: dict):

#         p_feats = []
#         for did in self.coords_depths:
#             vs = svh.grids[did].voxel_size
#             p = (xyz % vs) / vs - 0.5
#             p_feats.append(p)
#         p = torch.cat(p_feats, dim=1)

#         c_feats = []
#         for did in range(self.multiscale_depths):
#             if svh.grids[did] is None:
#                 c = torch.zeros((xyz.size(0), self.c_each_dim), device=xyz.device)
#             else:
#                 ijk_coords = svh.grids[did].active_grid_coords()
#                 coords = svh.grids[did].grid_to_world(ijk_coords.float()) # M, 3 voxel centers world coordinates
#                 # query_indices, _, _ = knn(torch.from_numpy(query_xyz).to(device), torch.from_numpy(voxel_center).to(device), 1)
#                 nn = NearestNeighbors(n_neighbors=self.k_neighbors)
#                 nn.fit(coords.cpu().numpy())  # coords is an (N, 3) array
#                 dist, indx = nn.kneighbors(xyz.cpu().numpy())  # xyz is an (M, 3) array
#                 indx = torch.from_numpy(indx).to(xyz.device)
#                 dist = torch.from_numpy(dist).to(xyz.device)
#                 gathered_latents = multiscale_feat[did][indx] #N, K, C
#                 c = self.att_pooling(gathered_latents).squeeze(1) #M, 1, hidden_dim
#                 # c = svh.grids[did].sample_trilinear(xyz, multiscale_feat[did])
#             c_feats.append(c)

#         if self.aggregation == 'cat':
#             c = torch.cat(c_feats, dim=1)
#         else:
#             c = sum(c_feats)

#         net = self.fc_p(p)
#         for i in range(self.n_blocks):
#             net = net + self.fc_c[i](c)
#             net = self.blocks[i](net)
#         out = self.fc_out(F.relu(net))

#         return out
