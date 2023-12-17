import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import functools

import numpy as np
from tqdm import tqdm

from pytorch3d.ops import knn_gather, knn_points

def compute_knn_feat(vsrc, vtar, vfeat, K=1):
    dist, idx, Vnn = knn_points(vsrc, vtar, K=K, return_nn=True)
    return knn_gather(vfeat, idx)

def homogenize(v, dim=2, is_direction=False):
    '''
    args:
        v: (B, N, C) or (B, C, N), depending on the specified dim
    return:
        (B, N, C+1)
    also supports channel first
    if is_direction=True --> a directional vector, the padded homo coord should have w=0, otherwise 1.
    '''
    pad_fn = (lambda x: torch.zeros_like(x)) if is_direction else (lambda x: torch.ones_like(x))

    if dim == 2:
        return torch.cat([v, pad_fn(v[:,:,:1])], -1)
    elif dim == 1:
        return torch.cat([v, pad_fn(v[:,:1,:])], 1)
    else:
        raise NotImplementedError('unsupported homogenize dimension [%d]' % dim)

def transform_normal(net, x, n):
    '''
    args:
        flow network that returns (B, 3, N)
        x: (B, N, 3)
        n: (B, N, 3)
    '''
    x = x.permute(0,2,1)
    with torch.enable_grad():
        x.requires_grad_()

        pred = net.query(x)

        dfdx = autograd.grad(
                [pred.sum()], [x], 
                create_graph=True, retain_graph=True, only_inputs=True)[0]
        print(dfdx.shape)
        # torch.einsum('bc')           
        #     if normalize:
        #         normal = F.normalize(normal, dim=1, eps=1e-6)

def get_posemap(map_type, n_joints, parents, n_traverse=1, normalize=True):
    pose_map = torch.zeros(n_joints,n_joints-1)
    if map_type == 'parent':
        for i in range(n_joints-1):
            pose_map[i+1,i] = 1.0
    elif map_type == 'children':
        for i in range(n_joints-1):
            parent = parents[i+1]
            for j in range(n_traverse):
                pose_map[parent, i] += 1.0
                if parent == 0:
                    break
                parent = parents[parent]
        if normalize:
            pose_map /= pose_map.sum(0,keepdim=True)+1e-16
    elif map_type == 'both':
        for i in range(n_joints-1):
            pose_map[i+1,i] += 1.0
            parent = parents[i+1]
            for j in range(n_traverse):
                pose_map[parent, i] += 1.0
                if parent != 0:
                    pose_map[i+1, parent-1] += 1.0
                if parent == 0:
                    break
                parent = parents[parent]
        if normalize:
            pose_map /= pose_map.sum(0,keepdim=True)+1e-16
    else:
        raise NotImplementedError('unsupported pose map type [%s]' % map_type)
    return pose_map

def batch_rot2euler(R):
    '''
    args:
        Rs: (B, 3, 3)
    return:
        (B, 3) euler angle (x, y, z)
    '''
    sy = torch.sqrt(R[:,0,0] * R[:,0,0] +  R[:,1,0] * R[:,1,0])
    singular = (sy < 1e-6).float()[:,None]

    x = torch.atan2(R[:,2,1] , R[:,2,2])
    y = torch.atan2(-R[:,2,0], sy)
    z = torch.atan2(R[:,1,0], R[:,0,0])
    euler = torch.stack([x,y,z],1)

    euler_s = euler.clone()
    euler_s[:,0] = torch.atan2(-R[:,1,2], R[:,1,1])
    euler_s[:,1] = torch.atan2(-R[:,2,0], sy)
    euler_s[:,2] = 0

    return (1.0-singular)*euler + singular * euler_s


def batch_rod2euler(rot_vecs):
    R = batch_rodrigues(rot_vecs)
    return batch_rot2euler(R)

def batch_rod2quat(rot_vecs):
    batch_size = rot_vecs.shape[0]

    angle = torch.norm(rot_vecs + 1e-16, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.cos(angle / 2)
    sin = torch.sin(angle / 2)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)

    qx = rx * sin
    qy = ry * sin
    qz = rz * sin
    qw = cos-1.0

    return torch.cat([qx,qy,qz,qw], dim=1)

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def quat_to_matrix(rvec):
    '''
    args:
        rvec: (B, N, 4)
    '''
    B, N, _ = rvec.size()

    theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=2))
    rvec = rvec / theta[:, :, None]
    return torch.stack((
        1. - 2. * rvec[:, :, 1] ** 2 - 2. * rvec[:, :, 2] ** 2,
        2. * (rvec[:, :, 0] * rvec[:, :, 1] - rvec[:, :, 2] * rvec[:, :, 3]),
        2. * (rvec[:, :, 0] * rvec[:, :, 2] + rvec[:, :, 1] * rvec[:, :, 3]),

        2. * (rvec[:, :, 0] * rvec[:, :, 1] + rvec[:, :, 2] * rvec[:, :, 3]),
        1. - 2. * rvec[:, :, 0] ** 2 - 2. * rvec[:, :, 2] ** 2,
        2. * (rvec[:, :, 1] * rvec[:, :, 2] - rvec[:, :, 0] * rvec[:, :, 3]),

        2. * (rvec[:, :, 0] * rvec[:, :, 2] - rvec[:, :, 1] * rvec[:, :, 3]),
        2. * (rvec[:, :, 0] * rvec[:, :, 3] + rvec[:, :, 1] * rvec[:, :, 2]),
        1. - 2. * rvec[:, :, 0] ** 2 - 2. * rvec[:, :, 1] ** 2
        ), dim=2).view(B, N, 3, 3)

def rot6d_to_matrix(rot6d):
    '''
    args:
        rot6d: (B, N, 6)
    return:
        rotation matrix: (B, N, 3, 3)
    '''
    x_raw = rot6d[:,:,0:3]
    y_raw = rot6d[:,:,3:6]
        
    x = F.normalize(x_raw, dim=2)
    z = torch.cross(x, y_raw, dim=2)
    z = F.normalize(z, dim=2)
    y = torch.cross(z, x, dim=2)
        
    rotmat = torch.cat((x[:,:,:,None],y[:,:,:,None],z[:,:,:,None]), -1) # (B, 3, 3)
    
    return rotmat

def compute_affinemat(param, rot_dim):
    '''
    args:
        param: (B, N, 9/12)
    return:
        (B, N, 4, 4)
    '''
    B, N, C = param.size()
    rot = param[:,:,:rot_dim]

    if C - rot_dim == 3:
        trans = param[:,:,rot_dim:]
        scale = torch.ones_like(trans)
    elif C - rot_dim == 6:
        trans = param[:,:,rot_dim:(rot_dim+3)]
        scale = param[:,:,(rot_dim+3):]
    else:
        raise ValueError('unsupported dimension [%d]' % C)
    
    if rot_dim == 3:
        rotmat = batch_rodrigues(rot)
    elif rot_dim == 4:
        rotmat = quat_to_matrix(rot)
    elif rot_dim == 6:
        rotmat = rot6d_to_matrix(rot)
    else:
        raise NotImplementedError('unsupported rot dimension [%d]' % rot_dim)
    
    A = torch.eye(4)[None,None].to(param.device).expand(B, N, -1, -1).contiguous()
    A[:,:,:3, 3] = trans # (B, N, 3, 1)
    A[:,:,:3,:3] = rotmat * scale[:,:,None,:] # (B, N, 3, 3)

    return A

def compositional_affine(param, num_comp, rot_dim):
    '''
    args:
        param: (B, N, M*(9/12)+M)
    return:
        (B, N, 4, 4)
    '''
    B, N, _ = param.size()

    weight = torch.exp(param[:,:,:num_comp])[:,:,:,None,None]

    affine_param = param[:,:,num_comp:].reshape(B, N*num_comp, -1)
    A = compute_affinemat(affine_param, rot_dim).view(B, N, num_comp, 4, 4)

    return (weight * A).sum(2) / weight.sum(dim=2).clamp(min=0.001)

