import torch
import torch.nn.functional as F
import numpy as np
import copy

import sys
from os.path import dirname, realpath, join
PROJECT_DIR = dirname(realpath(__file__))
sys.path.append(join(PROJECT_DIR,'..'))

from lib.utils_model import init_mlp_siren, init_net, get_embedder
from lib.mlp import MLP



class LBS_Net_Scanimate(torch.nn.Module):
    '''
    ! adapted from SCANimate
    '''
    def __init__(self,
                 opt,
                 ):
        super(LBS_Net_Scanimate, self).__init__()

        self.name = 'lbs_net'
        self.opt = copy.deepcopy(opt)

        body_model_type = self.opt['body_model_type']
        n_joints = 22 if body_model_type == 'smplx' else 24

        if opt['use_embed']:
            self.embedder, self.opt['mlp']['ch_dim'][0] = get_embedder(opt['d_size'], input_dims=opt['mlp']['ch_dim'][0])
        else:
            self.embedder = None

        if 'g_dim' in self.opt:
            self.opt['mlp']['ch_dim'][0] += self.opt['g_dim']
        if 'pose_dim' in self.opt:
            self.opt['mlp']['ch_dim'][0] += self.opt['pose_dim'] * 23
        if 'clo_feat_dim' in self.opt:
            self.opt['mlp']['ch_dim'][0] += self.opt['clo_feat_dim']
        self.opt['mlp']['ch_dim'][-1] = n_joints

        self.mlp = MLP(
            filter_channels=self.opt['mlp']['ch_dim'],
            res_layers=self.opt['mlp']['res_layers'],
            last_op=self.opt['mlp']['last_op'],
            nlactiv=self.opt['mlp']['nlactiv'],
            norm=self.opt['mlp']['norm'])

        init_net(self)

        if self.opt['mlp']['nlactiv'] == 'sin': # SIREN
            self.mlp.apply(init_mlp_siren)

        # self.register_buffer('bbox_min', torch.Tensor(bbox_min)[None,:,None])
        # self.register_buffer('bbox_max', torch.Tensor(bbox_max)[None,:,None])

        # self.feat3d = None
        self.global_feat = None

    def set_global_feat(self, feat):
        self.global_feat = feat

    def query(self, points, local_feat, normalize_coords=False, bmin=None, bmax=None):
        '''
        Given 3D points, query the network predictions for each point.
        args:
            points: (B, 3, N)
            local_feat: (B, n_dim, N), local features at the query points
        return:
            (B, C, N)
        '''
        N = points.size(2)
        
        if normalize_coords:
            if bmin is None:
                bmin = self.bbox_min
            if bmax is None:
                bmax = self.bbox_max
            points_nc3d = 2.0 * (points - bmin) / (bmax - bmin) - 1.0 # normalized coordiante
        else:
            points_nc3d = 1.0*points

        # if local_feat is not None:
        #     point_local_feat = index3d_custom(self.feat3d, points_nc3d)
        # else: # not body_centric_encoding
        #     point_local_feat = points_nc3d

        point_feat_cat = points_nc3d

        if self.embedder is not None:
            point_feat_cat = self.embedder(point_feat_cat.permute(0,2,1)).permute(0,2,1)

        if local_feat is not None:
            point_feat_cat = torch.cat([point_feat_cat, local_feat], 1) # [B, ndim, N]

        if self.global_feat is not None: #global_feat: [B, ndim]
            point_feat_cat = torch.cat([point_feat_cat, self.global_feat[:,:,None].expand(-1,-1,N)], 1)
        w0 = 30.0 if self.opt['mlp']['nlactiv'] == 'sin' else 1.0

        return self.mlp(w0*point_feat_cat)


    def forward(self, points, local_feat):
        return self.query(points, local_feat)
