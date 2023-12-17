'''
Diffuse the SMPL/SMPLX surface LBS weights to a grid in R3, and then apply smoothing to the grid
Smoothing can be done by either:
    - Gaussian smoothing (see below): applying a fixed Gaussian kernel to convolve the grid
    - Optimization: for each grid point, minimize the difference of its lbs weights to its neighbors, i.e. minimize the Laplacian
note: "diffusion" here is not to be confused with the generative diffusion model

Example usage:
    python lbs_grid_diffuse/optim_lbsw.py 2 128 5 5

'''

import numpy as np
import torch

import os
from os.path import join, basename, dirname, realpath
import sys
PROJECT_DIR = join(dirname(realpath(__file__)), '..')
sys.path.append(PROJECT_DIR)

from lib.modules import GridLaplacian, GaussianSmoothing
from diffuse_lbsw import DiffuseLBSW

pow = 2
grid_resl = 128
kernel_size = 5

debug = False
body_model = 'smpl'
thres = 0.00 # if query pt to smpl surface dist < thres, then this query pt doesn't get optimized
n_joints = 24 if body_model == 'smpl' else 22

mode = 'optimization' #'gaussian_smoothing' # or 'optimization'
os.makedirs(join(PROJECT_DIR, 'visualization', 'optimized_lbsw'), exist_ok=True)

diffuser = DiffuseLBSW(grid_resl=int(grid_resl), body_model_name=body_model, blend_weight_pow=int(pow))
diffuser.get_body_part_center()
centers = diffuser.bone_centers

from lib.utils_vis import color_lbsw
from lib.utils_io import customized_export_ply

nnbgr_grid_lbsw, dists = diffuser.nearest_neighbor_diffusion(query_pts=diffuser.grid, return_nearest_dists=True)

nnbghr_lbsw = nnbgr_grid_lbsw.reshape(grid_resl, grid_resl, -1, n_joints)
nnbghr_lbsw = torch.tensor(nnbghr_lbsw).unsqueeze(0).permute(0,4,1,2,3).float().cuda().contiguous()
incr_lbsw = torch.zeros_like(nnbghr_lbsw).float().cuda()
nnbghr_lbsw_init = nnbghr_lbsw.clone()

num_iters = 10 if mode == 'gaussian_smoothing' else 3001

if mode == 'gaussian_smoothing':
    print('-------Performing Gaussian smoothing to the nearest-neighbor diffused LBS weight field')
    gaussian_conv = GaussianSmoothing(channels=n_joints, kernel_size=kernel_size, dim=3)
    smoothed = {}
    smoothed[0] = nnbghr_lbsw

    grid_to_save = nnbghr_lbsw.squeeze(0).permute(1,2,3,0).reshape(-1, n_joints).detach().cpu()
    torch.save(grid_to_save.detach().float(), join(PROJECT_DIR, 'visualization', 'optimized_lbsw', 'gaussian_kernel{}_iter0_{}.pt'.format(kernel_size, body_model)))
    color = color_lbsw(grid_to_save, mode='diffuse', shuffle_color=True)
    customized_export_ply(join(PROJECT_DIR, 'visualization', 'optimized_lbsw', 'gaussian_kernel{}_iter0_{}.ply'.format(kernel_size, body_model)), diffuser.grid.cpu(), v_c=color)


    smoothed_grid = nnbghr_lbsw
    for iter in range(num_iters):
        print('Gaussian smoothing, iter {}'.format(iter+1))
        # kernel_size = 2*iter+3
        smoothed_grid = gaussian_conv(smoothed_grid)
        smoothed_grid = smoothed_grid.clamp(0.0, 1.0)
        smoothed_grid = smoothed_grid / smoothed_grid.norm(dim=1, p=1)
        smoothed[iter+1] = smoothed_grid

        grid_to_save = smoothed_grid.squeeze(0).permute(1,2,3,0).reshape(-1, n_joints).detach().cpu()
        torch.save(smoothed_grid.detach().float(), join(PROJECT_DIR, 'visualization', 'optimized_lbsw', 'gaussian_kernel{}_iter{}_{}.pt'.format(kernel_size, iter+1, body_model)))
        color = color_lbsw(grid_to_save, mode='diffuse', shuffle_color=True)
        customized_export_ply(join(PROJECT_DIR, 'visualization', 'optimized_lbsw', 'gaussian_kernel{}_iter{}_{}.ply'.format(kernel_size, iter+1, body_model)), diffuser.grid.cpu(), v_c=color)
        

if mode == 'optimization':
    print('-------Optimizing the LBS weight field with smoothness loss')
    incr_lbsw.requires_grad=True

    off_surface_pt_mask = dists > thres
    off_surface_pt_mask = torch.tensor(off_surface_pt_mask).float()
    off_surface_pt_mask = off_surface_pt_mask.reshape(grid_resl, grid_resl, -1).unsqueeze(0).unsqueeze(0).expand(-1, n_joints, -1, -1, -1)
    off_surface_pt_mask = off_surface_pt_mask.cuda()

    assert off_surface_pt_mask.shape == nnbghr_lbsw.shape

    lapconv = GridLaplacian(kernel_size=kernel_size, channels=n_joints, dim=3).cuda()

    # apply mask: surface points don't get grads
    incr_lbsw.register_hook(lambda grad: grad*off_surface_pt_mask)
    optimizer = torch.optim.Adam([incr_lbsw], lr=5e-4)

    loss_previous = 1e4
    for i in range(num_iters):
        optimizer.zero_grad()
        
        lbsw_now = incr_lbsw + nnbghr_lbsw
        lbsw_now = lbsw_now.clamp(0.0, 1.0)
        lbsw_now = lbsw_now / lbsw_now.norm(dim=1, p=1)

        lap_out = lapconv(lbsw_now)
        lap_norm = lap_out.norm(dim=1).mean()

        loss = lap_norm #+ deviation_norm

        if i%10 == 0:
            print(i, loss)

        if ((i%200 == 0) and (loss < loss_previous)):
            loss_previous = loss
            lbsw_now_normed = lbsw_now / lbsw_now.norm(dim=1, p=1)
            nnbghr_lbsw_final = lbsw_now_normed.squeeze(0).permute(1,2,3,0).reshape(-1, n_joints).detach().cpu()
            torch.save(lbsw_now_normed.detach().float(), join(PROJECT_DIR, 'visualization', 'optimized_lbsw', 'optimized_lbsw_kernel{}_step{}_{}.pt'.format(kernel_size, i, body_model)))
            color = color_lbsw(nnbghr_lbsw_final, mode='diffuse', shuffle_color=False)
            customized_export_ply(join(PROJECT_DIR, 'visualization', 'optimized_lbsw', 'optimized_lbsw_kernel{}_step{}_{}.ply'.format(kernel_size, i, body_model)), diffuser.grid.cpu(), v_c=color)
        
        loss.backward()
        optimizer.step()