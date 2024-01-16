# render object
from diff_gaussian_rasterization_object import GaussianRasterizationSettings as Object_GaussianRasterizationSettings
from diff_gaussian_rasterization_object import GaussianRasterizer as Object_GaussianRasterizer

# render depth (kiui)
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings as Depth_GaussianRasterizationSettings
from diff_gaussian_rasterization_depth import GaussianRasterizer as Depth_GaussianRasterizer

# render depth & silhouette
from diff_gaussian_rasterization import GaussianRasterizationSettings as Silhouette_GaussianRasterizationSettings
from diff_gaussian_rasterization import GaussianRasterizer as Silhouette_GaussianRasterizer

import torch
import torch.nn.functional as F


# render rendered_image, radii, objects
def render_object(params, transformed_pts, cam_setup):
    rasterizer = Object_GaussianRasterizer(raster_settings=cam_setup)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_objects = rasterizer(
        means3D = transformed_pts,
        means2D = torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0,
        # shs = shs,
        sh_objs = params["obj_dc"],
        colors_precomp = params['rgb_colors'],
        opacities = torch.sigmoid(params['logit_opacities']),
        scales = torch.exp(torch.tile(params['log_scales'], (1, 3))),
        rotations = F.normalize(params['unnorm_rotations']))
    
    return rendered_objects


def render_image(params, transformed_pts, cam_setup):

    rasterizer = Depth_GaussianRasterizer(raster_settings=cam_setup)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
            # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(

        means3D = transformed_pts,
        means2D = torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0,
        # shs=shs,
        colors_precomp =  params['rgb_colors'],
        opacities = torch.sigmoid(params['logit_opacities']),
        scales = torch.exp(torch.tile(params['log_scales'], (1, 3))),
        rotations = F.normalize(params['unnorm_rotations'])),
        # cov3D_precomp=cov3D_precomp,
    )

    rendered_image = rendered_image.clamp(0, 1)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "image": rendered_image,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }



def render_silhouette(params, time_idx, cam_setup):
    
    pass