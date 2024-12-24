#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from einops import repeat
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel


def compare_two_masks(mask1, mask2):
    return_idx = []
    mask1_true_idx = torch.where(mask1)[0]
    mask2_false_idx = torch.where(~mask2)[0]

    for idx2 in mask2_false_idx:
        if idx2 in mask1_true_idx:
            idx1 = (mask1_true_idx == idx2).nonzero().squeeze(0)
            return_idx.append(idx1)

    return torch.cat(return_idx, dim=0)


def generate_neural_gaussians(viewpoint_camera, pc: GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)

    if pc.extra_mask is not None:
        HRLVI_mask = compare_two_masks(visible_mask, pc.extra_mask)
    else:
        HRLVI_mask = None

    # before visible_mask: 53005 anchors; after visible_mask: 18465 anchors
    feat = pc._anchor_feat[visible_mask]  # 18465x32
    anchor = pc.get_anchor[visible_mask]  # 18465x3
    grid_offsets = pc._offset[visible_mask]  # 18465x10x3
    grid_scaling = pc.get_scaling[visible_mask]  # 18465x6

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center  # 18465x3
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)  # 18465x1
    # view
    ob_view = ob_view / ob_dist  # 18465x3

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)  # nx32x1
        feat = feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1] + \
            feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2] + \
            feat[:, ::1, :1] * bank_weight[:, :, 2:]  # nx32x1
        feat = feat.squeeze(dim=-1)  # [n, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # 18465x(32+3+1)  [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]

    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:, 0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)  # 18465x10

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])  # 184650x1
    mask = (neural_opacity > 0.0)  # 184650x1
    mask = mask.view(-1)  # [184650]

    # select opacity 
    opacity = neural_opacity[mask]  # 110790x1

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)  # 18465x30
    if HRLVI_mask is None:
        color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])  # 184650x3 [mask]
    else:
        color = color.reshape([anchor.shape[0], pc.n_offsets, 3])
        color[HRLVI_mask] = torch.tensor([0.0, 1.0, 0.0], device="cuda")
        color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)  # 18465x70
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7])  # 184650x7 [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3])  # 184650x3 [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [18465x6, 18465x3] --> 18465x9
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # 184650x9
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)  # 184650x22
    masked = concatenated_all[mask]  # 110790x22
    # scaling_repeat:110790x6 | repeat_anchor:110790x3 | color:110790x3 | scale_rot:110790x7 | offsets:110790x3
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])  # 110790x3 * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:, 3:7])  # 110790x4
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:, :3]  # 110790x3
    xyz = repeat_anchor + offsets  # 110790x3

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, visible_mask=None, retain_grad=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.DEBUG
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "depth": depth
                }


def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.DEBUG
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.COMPUTE_COV3D_PYTHON:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    return radii_pure > 0
