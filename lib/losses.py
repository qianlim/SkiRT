from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def chamfer_loss_separate(output, target, weight=1e4, phase='train', debug=False):
    from chamferdist.chamferdist import ChamferDistance
    cdist = ChamferDistance()
    model2scan, scan2model, idx1, idx2 = cdist(output, target)
    if phase == 'train':
        return model2scan, scan2model, idx1, idx2
    else: # in test, show both directions, average over points, but keep batch
        return torch.mean(model2scan, dim=-1)* weight, torch.mean(scan2model, dim=-1)* weight,


def normal_loss(output_normals, target_normals, nearest_idx, weight=1.0, mask=None, phase='train'):
    '''
    Given the set of nearest neighbors found by chamfer distance, calculate the
    L1 discrepancy between the predicted and GT normals on each nearest neighbor point pairs.
    Note: the input normals are already normalized (length==1).
    
    mask: [num_output_points], a binary [0,1] mask indicating which points from the output pcl should be excluded from loss computation
    '''
    nearest_idx = nearest_idx.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
    target_normals_chosen = torch.gather(target_normals, dim=1, index=nearest_idx)

    assert output_normals.shape == target_normals_chosen.shape

    if phase == 'train':
        if mask is None:
            lnormal = F.l1_loss(output_normals, target_normals_chosen, reduction='mean')  # [batch, 8000, 3])
        else:
            lnormal = F.l1_loss(output_normals, target_normals_chosen, reduction='none')  # [batch, 8000, 3])
            lnormal = lnormal * mask.unsqueeze(0).unsqueeze(-1)
            lnormal = lnormal.mean()
        return lnormal, target_normals_chosen
    else:
        lnormal = F.l1_loss(output_normals, target_normals_chosen, reduction='none')
        lnormal = lnormal.mean(-1).mean(-1) # avg over all but batch axis
        return lnormal, target_normals_chosen


def color_loss(output_colors, target_colors, nearest_idx, weight=1.0, phase='train', excl_holes=False):
    '''
    Similar to normal loss, used in training a color prediction model.
    '''
    nearest_idx = nearest_idx.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
    target_colors_chosen = torch.gather(target_colors, dim=1, index=nearest_idx)

    assert output_colors.shape == target_colors_chosen.shape
    
    if excl_holes:
        # scan holes have rgb all=0, exclude these from supervision
        colorsum = target_colors_chosen.sum(-1)
        mask = (colorsum!=0).float().unsqueeze(-1)
    else:
        mask = 1.

    if phase == 'train':
        lcolor = F.l1_loss(output_colors, target_colors_chosen, reduction='none')  # [batch, 8000, 3])
        lcolor = lcolor * mask
        lcolor = lcolor.mean()
        return lcolor, target_colors_chosen
    else:
        lcolor = F.l1_loss(output_colors, target_colors_chosen, reduction='none')
        lcolor = lcolor * mask
        lcolor = lcolor.mean(-1).mean(-1) # avg over all but batch axis
        return lcolor, target_colors_chosen



''' the TV loss next is directly copied from Kornia library source code'''

def total_variation(img: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Total Variation according to [1].

    Args:
        img (torch.Tensor): the input image with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.

    Return:
        torch.Tensor: a scalar with the computer loss.

    Examples:
        >>> total_variation(torch.ones(3, 4, 4))
        tensor(0.)

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

    if len(img.shape) < 3 or len(img.shape) > 4:
        raise ValueError(
            f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}."
        )

    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :] # right pixel - left pixel
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1] # lower pixel - upper pixel

    reduce_axes = (-3, -2, -1)
    res1 = pixel_dif1.abs().sum(dim=reduce_axes)
    res2 = pixel_dif2.abs().sum(dim=reduce_axes)

    return res1 + res2

def total_variation_masked(img: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Total Variation according to [1].

    Args:
        img (torch.Tensor): the input image with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        boundary_mask (torch.Tensor): (N, 1, H, W) or (1, H, W), binary (0/1) mask for if a pixel is a valid pixel for computing the TV loss

        NOTE: a pixel is 'valid' if:
            - all its neigbhors are in a uv island
            - it's on the uv island boundary, but both its left neighbor AND upper neighbor is not black (invalid) pixel

    Return:
        torch.Tensor: a scalar with the computer loss.

    Examples:
        >>> total_variation(torch.ones(3, 4, 4))
        tensor(0.)

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

    if len(img.shape) < 3 or len(img.shape) > 4:
        raise ValueError(
            f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}."
        )

    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    reduce_axes = (-3, -2, -1)
    
    res1 = pixel_dif1.abs().sum(dim=reduce_axes)
    res2 = pixel_dif2.abs().sum(dim=reduce_axes)

    return res1 + res2


def uniformity_metric(pcl, k=5, return_mean=True):
    '''
    pcl: [B, N, 3]

    get the stats (max, median, mean, std) of each point's average knn distance (excluding to itself) in a point cloud
    '''
    from pytorch3d.ops import knn_points
    import torch
    knn_out = knn_points(pcl, pcl, K=k)
    dists, idx = knn_out[0], knn_out[1]
    dists_k_avg = torch.sqrt(dists)[...,1:].mean(-1) # dists returned by pytorch3d knn are squared; [1:] removes the nearest ngbr (a point itself)
    
    d_mean = dists_k_avg.mean(-1)
    d_std = dists_k_avg.std(-1)
    d_med = dists_k_avg.median(-1)[0]
    d_max = dists_k_avg.max(-1)[0]
    
    results = d_mean, d_std, d_med, d_max
    if return_mean:
        return [x.mean(0) for x in results]
    else:
        return results
    