import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True

from lib.utils_model import PositionalEncoding, normalize_uv, uv_to_grid, init_mlp_siren
from lib.modules import UnetNoCond5DS, UnetNoCond6DS, UnetNoCond7DS, ShapeDecoderDeep, PoseEmbNet
from lib.lbs_net import LBS_Net_Scanimate
from lib.utils_pose import homogenize


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            if k in ['pose_map', 'body_lbsw']:
                v = v.cuda()
            setattr(self, k, v)

        self.actv_fn = self.shape_mlp_opt['actv_fn']

        self.register_buffer('cano_vt', torch.tensor(self.smpl_cano_vt).float())
        # self.register_buffer('cano_vnormal', torch.tensor(self.smpl_cano_vnormal).float())
        self.cano_vnormal = torch.tensor(self.smpl_cano_vnormal).float().cuda()

        self.register_buffer('faces', torch.tensor(self.smpl_f).long())
        # self.register_buffer('verts_uv', torch.tensor(self.smpl_v_uv).float())
        self.verts_uv = torch.tensor(self.smpl_v_uv).float().cuda()
        # self.register_buffer('faces_uv', torch.tensor(self.smpl_f_uv).long())
        self.faces_uv = torch.tensor(self.smpl_f_uv).long().cuda()

        self.num_joints = self.body_lbsw.shape[1] - 1

        if self.pos_encoding:
            self.embedder = PositionalEncoding(num_freqs=self.num_emb_freqs,
                                               input_dims=self.query_coord_dim,
                                               include_input=self.posemb_incl_input)
            self.embedder.create_embedding_fn()
            self.query_coord_dim = self.embedder.out_dim

        if self.incl_query_nml: # add normals of the query points (at the minimally-clothed body surface) as input
            self.query_coord_dim = self.query_coord_dim + 3


    def interp_geom_feats(self, geom_featmap, query_grid=None, bary=None, face_idx=None, num_query_pts=None, interp_same_bary_idx=True, align_corners=False):
        if not self.use_global_geom_feat:
            if not self.use_vert_geom_feat:
                geom_featmap = F.grid_sample(geom_featmap, query_grid, mode='bilinear', align_corners=align_corners)#, align_corners=True)
                geom_featmap = geom_featmap.squeeze(-1)
            else:
                geom_featmap = self.interp_features_bary(geom_featmap, bary, face_idx.squeeze(0), interp_same_bary_idx=interp_same_bary_idx)
        else:
            geom_featmap = geom_featmap.expand(-1, -1, num_query_pts)
        
        return geom_featmap


    def normalize_query_pts(self, query_pts, center, sc_factor):
        return (query_pts - center) * sc_factor


    def encode_query_pts(self, uv_loc=None, query_loc_3d=None, query_loc_nml=None, valid_idx=None):
        B = query_loc_3d.shape[0]

        if self.pos_encoding:
            if not self.query_xyz:
                uv_loc = normalize_uv(uv_loc).view(-1, uv_loc.shape[-1])
                uv_loc = self.embedder.embed(uv_loc).view(B, -1,self.embedder.out_dim)
            else:
                query_loc_3d = self.embedder.embed(query_loc_3d.view(-1, 3)).view(B, -1, self.embedder.out_dim) #.transpose(1,2).contiguous()
                uv_loc = uv_loc.transpose(1,2) # just for backwards compatibility
        else:
            uv_loc = uv_loc.reshape(B, -1, 2).transpose(1, 2)

        if valid_idx is not None:
            uv_loc = uv_loc[...,valid_idx]


        if self.query_xyz:
            query_locs = query_loc_3d.transpose(2,1).contiguous() # channel last --> channel second-to-last
            query_nml = query_loc_nml.transpose(2,1).contiguous()
            if self.incl_query_nml:
                query_locs = torch.cat([query_locs, query_nml],1)
        else:
            query_locs = uv_loc
        
        return query_locs

    def get_pose_feat_lbs(self, lbsw_local, pose_params):
        '''
        get pose feature following the style of SCANimate Eq. 8:
        at query location x, the pose_paramsure(x) = matmul(pose_map, lbsw(x))* pose_params


        lbsw_local: [B, n_joints, n_points], linear blend skinning weights at each query location
        pose_params: [B, n_joints, C_pose] C can be 3 for axis-angle, 4 for quarternion, 9 for rot matrix

        out:
            (B, pose_feat_dim, N)

        '''
        B, _, N = lbsw_local.shape

        lbsw_filtered = torch.einsum('bjv,jl->blv', lbsw_local, self.pose_map) # [B, num_joints-1, num_points]
        lbsw_filtered = lbsw_filtered.contiguous()

        pose_params_filtered = pose_params.unsqueeze(-1) * lbsw_filtered.unsqueeze(-2) # shape: [B, nun_joints-1, 4, 1] x [B, nun_joints-1, 1, num_points], -1 because removed root
        pose_params_filtered = pose_params_filtered.reshape(B, -1, N) # [B, (n_joints*pose feat dim per joint), num_points]

        return lbsw_filtered, pose_params_filtered


    def interp_features_bary(self, vert_features, bary_coords, sample_face_idxs, interp_same_bary_idx=True):
        '''
        use barycentric interpolation to diffuse the features (aka functions) that
        are defined on the minimal body mesh vertices to the entire body surface

        if same_bary_idx:
            ! applicable to the same set of barycentric coords is shared for all meshes in the batch
        else:
            supports different bary coords and sample face idxs for each example within a batch

        ! assume all body meshes in the training have same topology (= our use case)
        
        the vertex features can be:
            - 3D location of the vertices
            - normal vectors defined on vertices
            - skinning weights
            - local clothing geometry features
        
        vert_features: [B, feat_dim, n_verts]

        if same_bary_idx:
            bary_coords: [1, n_points, 3], barycentric points of the sampled points
            sample_face_idxs: [n_points], indices of the triangles from the sampled mesh
        else:
            bary_coords: [B, n_points, 3], barycentric points of the sampled points, can be different per example in a batch.
            sample_face_idxs: [B, n_points], indices of the triangles from the sampled mesh. Can be different for each example in a batch.

        returns:
            interped_features: [B, feat_dim, n_points]
        '''
        if interp_same_bary_idx:
            B, feat_dim, _ = vert_features.shape 
            n_points = bary_coords.shape[1]
            if len(sample_face_idxs.shape) > 1:
                sample_face_idxs = sample_face_idxs.squeeze(0)

            interped_features = torch.zeros([B, feat_dim, n_points])

            w0, w1, w2 = bary_coords[...,0], bary_coords[...,1], bary_coords[...,2]
            
            vert_features = vert_features.permute([0,2,1]) # [B, n_verts, feat_dim]

            tris = vert_features[:, self.faces, :] # [B, n_faces, 3, feat_dim], vert features arranged in triangles

            tris = tris.view(-1, 3, feat_dim)

            feat_v0, feat_v1, feat_v2 = tris[:, 0, :], tris[:, 1, :], tris[:, 2, :] # [B*n_faces, feat_dim]
            # get feature on the selected (sampled) triangles' verts
            feat_v0_sel = feat_v0.reshape(B, -1, feat_dim)[:, sample_face_idxs, :]  # [B, n_points, feat_dim]
            feat_v1_sel = feat_v1.reshape(B, -1, feat_dim)[:, sample_face_idxs, :] 
            feat_v2_sel = feat_v2.reshape(B, -1, feat_dim)[:, sample_face_idxs, :] 

            interped_features = w0[:, :, None] * feat_v0_sel + w1[:, :, None] * feat_v1_sel + w2[:, :, None] * feat_v2_sel
        else:
            B, C, Nv = vert_features.shape
            faces_list = self.faces.expand(B, -1, -1)
            faces_packed = torch.cat([i*Nv + x for i, x in enumerate(faces_list)], dim=0) # a tensor of faces, i.e. *vertex indices*
                        
            vert_features = vert_features.permute(0,2,1).reshape(-1, C) #[B, Nv, C]

            num_meshes = len(vert_features)
            # Initialize samples tensor with fill value 0 for empty meshes.
            num_samples = bary_coords.shape[1]
            samples = torch.zeros((num_meshes, num_samples, 3), device=vert_features.device)
            
            # Get the vertex coordinates of the sampled faces.
            face_verts = vert_features[faces_packed]
            v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

            # Randomly generate barycentric coords.
            w0, w1, w2 = bary_coords[...,0], bary_coords[...,1], bary_coords[...,2]

            # Use the barycentric coords to get a point on each sampled face.
            a = v0[sample_face_idxs]  # (N, num_samples, 3) # note sample_face_idxs are a tensor of *face* indices, not to confuse with vertex indices above
            b = v1[sample_face_idxs]
            c = v2[sample_face_idxs]
            interped_features = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

        return interped_features.permute(0,2,1).contiguous()

    
    def coord_transf(self, pred_res, pred_normals, posed_pt_minimal, cano_subj_bs_v, jT, pred_lbsw,
                    local_vtransf=None, bary=None, face_idx=None, interp_same_bary_idx=True,
                    transf_scaling=1.0):
        '''
        local to global transform
        '''

        B = pred_res.shape[0]
        cano_pt_minimal = self.interp_features_bary(cano_subj_bs_v.transpose(2,1), bary, face_idx.squeeze(), interp_same_bary_idx=interp_same_bary_idx)
        pred_clo_cano = cano_pt_minimal + pred_res * transf_scaling
        
        pred_normals_cano = pred_normals.transpose(2,1)
        pred_normals_cano = F.normalize(pred_normals_cano, dim=-1)

        if not self.use_jT:
            pred_res = pred_res.permute([0,2,1]).unsqueeze(-1)
            pred_normals = pred_normals.permute([0,2,1]).unsqueeze(-1)

            pred_res = torch.matmul(local_vtransf, pred_res).squeeze(-1)
            pred_normals = torch.matmul(local_vtransf, pred_normals).squeeze(-1)
            pred_normals = F.normalize(pred_normals, dim=-1)

            # residual to abosolute locations in space
            pred_clo_posed = pred_res + posed_pt_minimal

            # reshaping the points that are grouped into patches into a big point set
            pred_clo_posed = pred_clo_posed.reshape(B, -1, 3).contiguous()
            pred_normals = pred_normals.reshape(B, -1, 3).contiguous()

            
        else:
            pred_T = torch.einsum('bjst,bjv->bvst', jT, pred_lbsw) 
            pred_T[:,:,3,3] = 1.0

            if self.transf_only_disp:
                pred_res = torch.einsum('bvst,btv->bvs', pred_T, homogenize(pred_res, 1, is_direction=True))[:,:,:3]
                pred_res = pred_res * transf_scaling
                pred_clo_posed = pred_res + posed_pt_minimal
            
            else:
                pred_res = torch.einsum('bvst,btv->bvs', pred_T, homogenize(pred_res, 1, is_direction=True))[:,:,:3]
                pred_res = pred_res * transf_scaling
                pred_body_posed = torch.einsum('bvst,btv->bvs', pred_T, homogenize(cano_pt_minimal, 1, is_direction=False))[:,:,:3]
                pred_clo_posed = pred_res + pred_body_posed

            pred_normals = torch.einsum('bvst,btv->bvs', pred_T, homogenize(pred_normals, 1, is_direction=True))[:,:,:3]
            pred_normals = F.normalize(pred_normals, dim=-1)


        return pred_clo_cano, pred_normals_cano, pred_clo_posed, pred_normals



class SkiRT_fine(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        unets = {32: UnetNoCond5DS, 64: UnetNoCond6DS, 128: UnetNoCond7DS, 256: UnetNoCond7DS}
        unet_loaded = unets[self.inp_posmap_size]

        if ((self.pose_feat_type in ['conv', 'both']) or (self.conv_only_cano_body)):
            # U-net: for extracting pixel-aligned pose features from the input UV positional maps
            self.unet_posefeat = unet_loaded(self.input_nc, self.c_pose, self.nf, up_mode=self.up_mode, use_dropout=self.use_dropout)
        
        if self.use_pose_emb:
            self.pose_emb_net = PoseEmbNet(in_size=self.num_joints*4, out_size=self.c_pose)

        if self.pose_feat_type.lower() in ['scanimate', 'both']:
            pose_filtering_dim = self.num_joints * 4 # num_joints * dim_pose_param per joint
            if ((self.pose_feat_type.lower() =='scanimate') and (not self.conv_only_cano_body)):
                 if not self.use_pose_emb:
                    self.c_pose = pose_filtering_dim
            else:
                if not self.use_pose_emb:
                    self.c_pose += self.num_joints * 4
                else:
                    self.c_pose += self.c_pose

        if self.use_pred_lbsw:
            self.lbs_net_opt['clo_feat_dim'] = self.c_geom
            self.lbs_net_opt['mlp']['ch_dim'][0] = self.query_coord_dim if not self.incl_query_nml else self.query_coord_dim - 3
            self.lbs_net = LBS_Net_Scanimate(opt=self.lbs_net_opt)

        print('\n==========pose feat dim {}, geom feat dim {}, query loc+nml dim {}, query normals: {}\n'.format(self.c_pose, self.c_geom,self.query_coord_dim, self.incl_query_nml))
        shape_decoder_in_size = self.query_coord_dim + self.c_geom + self.c_pose
        # shape_decoder_in_size = self.query_coord_dim if self.use_layer_cond else self.query_coord_dim + self.c_geom + self.c_pose
        self.shape_mlp_opt['in_size'] = shape_decoder_in_size
        self.decoder = ShapeDecoderDeep(**self.shape_mlp_opt)

        if self.shape_mlp_opt['actv_fn'] == 'sin':
            self.decoder.apply(init_mlp_siren)

    def forward(self, x, geom_featmap, 
                query_uv=None, query_bary=None, query_face_idx=None,
                jT=None, pose_params=None, 
                transf_scaling=1.0,
                cano_subj_bs_v=None, 
                posed_subj_v=None,
                coarse_clo_pts=None,
                pre_diffused_lbsw=None,
                interp_same_bary_idx=True,
                align_corners=False,
                non_handfeet_mask=None,
                train_progress=0.,):
        '''
        :param x: input posmap, [batch, 3, 256, 256]
        :param geom_featmap: a [B, C, H, W] tensor, spatially pixel-aligned with the pose features extracted by the UNet
        :param uv_loc: querying uv coordinates, ranging between 0 and 1, of shape [B, H*W, 2].

        train_progress: used to blend ease-in the predicted lbsw from the gt lbsw
        :return:
            clothing offset vectors (residuals) and normals of the points
        '''
        # get query points and body basis points (to be added to)
        B, num_query_pts, _, _ = query_uv.shape
        query_uv = normalize_uv(query_uv) # [B, N, 1, 2], required by grid sample
        vert_feats = torch.cat([self.cano_vt.expand(B, -1, -1).transpose(1,2), # 3
                                self.cano_vnormal.expand(B, -1, -1).transpose(1,2), # 3
                                cano_subj_bs_v.transpose(1,2), # 3
                                posed_subj_v.transpose(1,2),# 3
                                self.body_lbsw.expand(B, -1, -1)], # J
                                dim= 1) # [B, 3, N] or [B, J, N] for lbsw
        
        point_feats = self.interp_features_bary(vert_feats, query_bary, query_face_idx, interp_same_bary_idx)
        query_loc_3d, query_loc_nml, cano_pt_minimal, posed_pt_minimal, smpl_lbsw_local = point_feats[:, :3], point_feats[:, 3:6], point_feats[:, 6:9], point_feats[:, 9:12], point_feats[:, 12:]

        if ((coarse_clo_pts is not None) and (pre_diffused_lbsw is not None)):
            # cano_pt_minimal = coarse_clo_pts
            smpl_lbsw_local = pre_diffused_lbsw

        '''
        geometry features
        '''
        geom_featmap = self.interp_geom_feats(geom_featmap, query_uv, query_bary, query_face_idx,
                                             num_query_pts, interp_same_bary_idx=interp_same_bary_idx, align_corners=align_corners)
        '''
        pose features
        '''
        pose_featmaps = []

        if ((self.pose_feat_type.lower() in ['conv', 'both']) or (self.conv_only_cano_body)):
            # pose features
            pose_featmap_conv = self.unet_posefeat(x) # [B, C, H, W], h, W are lower-res than querying, e.g. 128
            
            pose_featmap_conv = F.grid_sample(pose_featmap_conv, query_uv, mode='bilinear', align_corners=align_corners) # [B, C, N, 1]
            pose_featmap_conv = pose_featmap_conv.squeeze(-1) # [B, C, N]
            query_uv = query_uv.squeeze(2).transpose(1,2)

            pose_featmaps.append(pose_featmap_conv)


        if self.pose_feat_type.lower() in ['scanimate', 'both']:
            # use scanimate-style pose-param filtering mechanism as pose feature
            _, pose_featmap_filtering = self.get_pose_feat_lbs(smpl_lbsw_local, pose_params)
            # if self.zero_pose_params:
            #     pose_featmap_filtering = torch.zeros_like(pose_featmap_filtering).cuda()
            if self.use_pose_emb:
                pose_featmap_filtering = self.pose_emb_net(pose_featmap_filtering)
            pose_featmaps.append(pose_featmap_filtering)

        pose_featmap = torch.cat(pose_featmaps, dim=1)

        '''
        LBSNet
        '''
        query_locs = query_loc_3d if self.query_xyz else query_uv # [B, 3 or 2, N]
        pred_lbsw = self.lbs_net(query_locs, geom_featmap) if self.use_pred_lbsw else smpl_lbsw_local

        if self.gradual_pred_lbsw: # in the beginning mostly use gt smpl lbsw, each epoch increase the portion of pred lbsw
            pred_lbsw = smpl_lbsw_local * (1-train_progress) + pred_lbsw * train_progress

        '''
        positional encoding and get valid points (if use UV map)
        '''
        if self.pos_encoding:
            query_dim = 3 if self.query_xyz else 2
            query_locs = query_locs.transpose(1,2).reshape(-1, query_dim)
            query_locs = self.embedder.embed(query_locs).view(B, -1, self.embedder.out_dim)
            query_locs = query_locs.transpose(1, 2).contiguous() # TODO check shape [B, 2, N]

        pix_feature = torch.cat([pose_featmap, geom_featmap], 1) # [B, C, N]

        if self.incl_query_nml:
            query_locs = torch.cat([query_locs, query_loc_nml], 1) # [B, C, N]

        w0 = 30.0 if self.actv_fn == 'sin' else 1.0
        net_input = torch.cat([pix_feature, query_locs], 1)

        '''
        decoder
        '''
        pred_res, pred_normals = self.decoder(net_input, local_feat_gain=w0)

        if non_handfeet_mask is not None:
            pred_res = pred_res * non_handfeet_mask.unsqueeze(0).unsqueeze(0)

        pred_res_cano = pred_res.clone().permute(0,2,1)

        '''
        local to global transform
        '''
        pred_clo_cano = cano_pt_minimal + pred_res * transf_scaling
        
        pred_normals_cano = pred_normals.transpose(2,1)
        pred_normals_cano = F.normalize(pred_normals_cano, dim=-1)

        pred_T = torch.einsum('bjst,bjv->bvst', jT, pred_lbsw) 
        pred_T[:,:,3,3] = 1.0

        pred_res = torch.einsum('bvst,btv->bvs', pred_T, homogenize(pred_res, 1, is_direction=True))[:,:,:3]
        pred_res = pred_res * transf_scaling
        pred_body_posed = torch.einsum('bvst,btv->bvs', pred_T, homogenize(cano_pt_minimal, 1, is_direction=False))[:,:,:3]
        pred_body_nml_posed = torch.einsum('bvst,btv->bvs', pred_T, homogenize(query_loc_nml, 1, is_direction=True))[:,:,:3]
        if self.transf_only_disp:
            if coarse_clo_pts is not None:
                pred_clo_posed = pred_res + coarse_clo_pts.transpose(2,1)
            else:
                pred_clo_posed = pred_res + posed_pt_minimal.transpose(2,1)
        else:
            if coarse_clo_pts is not None:
                pred_coarse_posed = torch.einsum('bvst,btv->bvs', pred_T, homogenize(coarse_clo_pts, 1, is_direction=False))[:,:,:3]
                pred_clo_posed = pred_res + pred_coarse_posed
            else:
                pred_clo_posed = pred_res + pred_body_posed

        pred_normals = torch.einsum('bvst,btv->bvs', pred_T, homogenize(pred_normals, 1, is_direction=True))[:,:,:3]
        pred_normals = F.normalize(pred_normals, dim=-1)

        if non_handfeet_mask is not None:
            hand_feet_pt_idx = (non_handfeet_mask !=1)
            pred_normals[:, hand_feet_pt_idx, :] = pred_body_nml_posed[:, hand_feet_pt_idx, :]
        preds = {
            'normals_posed': pred_normals,
            'normals_cano': pred_normals_cano,
            'disps_posed': pred_res,
            'disps_cano': pred_res_cano,
            'clothed_posed': pred_clo_posed,
            'clothed_cano': pred_clo_cano.transpose(2,1),
            'pred_body_posed': pred_body_posed,
            'pred_body_nml_posed': pred_body_nml_posed,
            'gt_body_posed': posed_pt_minimal.transpose(2,1),
            'pred_lbsw': pred_lbsw,
            'gt_lbsw': smpl_lbsw_local,
        } # all have shape [B, num_pts, 3], lbsw has shape [B, num_joints, num_pts]
        return preds



class SkiRT_Coarse(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print('\n==========Coarse shape network, geom feat dim {}, query loc+nml dim {}, query normals: {}\n'.format(self.c_geom,self.query_coord_dim, self.incl_query_nml))
        shape_decoder_in_size = self.query_coord_dim + self.c_geom
        self.shape_mlp_opt['in_size'] = shape_decoder_in_size
        self.decoder = ShapeDecoderDeep(**self.shape_mlp_opt)

        if self.shape_mlp_opt['actv_fn'] == 'sin': # SIREN # TODO for other actv_fn, also try the init_net from scanimate
            self.decoder.apply(init_mlp_siren)

    def forward(self, x, geom_featmap, 
                query_uv=None, query_bary=None, query_face_idx=None,
                jT=None, pose_params=None, 
                transf_scaling=1.0,
                cano_subj_bs_v=None, 
                posed_subj_v=None,
                interp_same_bary_idx=True,
                align_corners=False,
                non_handfeet_mask=None,
                coarse_clo_pts=None,
                pre_diffused_lbsw=None,
                train_progress=0.,
                ):
        '''
        :param x: input posmap, [batch, 3, 256, 256]
        :param geom_featmap: a [B, C, H, W] tensor, spatially pixel-aligned with the pose features extracted by the UNet
        :param uv_loc: querying uv coordinates, ranging between 0 and 1, of shape [B, H*W, 2].
        :return:
            clothing offset vectors (residuals) and normals of the points
        '''
        # get query points and body basis points (to be added to)
        B, num_query_pts, _, _ = query_uv.shape
        query_uv = normalize_uv(query_uv) # [B, N, 1, 2], required by grid sample
        vert_feats = torch.cat([self.cano_vt.expand(B, -1, -1).transpose(1,2), # 3
                                self.cano_vnormal.expand(B, -1, -1).transpose(1,2), # 3
                                cano_subj_bs_v.transpose(1,2), # 3
                                posed_subj_v.transpose(1,2),# 3
                                self.body_lbsw.expand(B, -1, -1)], # J
                                dim= 1) # [B, 3, N] or [B, J, N] for lbsw

        point_feats = self.interp_features_bary(vert_feats, query_bary, query_face_idx, interp_same_bary_idx)
        query_loc_3d, query_loc_nml, cano_pt_minimal, posed_pt_minimal, smpl_lbsw_local = point_feats[:, :3], point_feats[:, 3:6], point_feats[:, 6:9], point_feats[:, 9:12], point_feats[:, 12:]
        query_uv = query_uv.squeeze(2).transpose(1,2)

        query_locs = query_loc_3d if self.query_xyz else query_uv 
        
        '''
        geometry features
        '''
        geom_featmap = self.interp_geom_feats(geom_featmap, query_uv, query_bary, query_face_idx,
                                             num_query_pts, interp_same_bary_idx=interp_same_bary_idx, align_corners=align_corners)

        '''
        positional encoding and get valid points (if use UV map)
        '''
        if self.pos_encoding:
            query_dim = 3 if self.query_xyz else 2
            query_locs = query_locs.transpose(1,2).reshape(-1, query_dim)
            query_locs = self.embedder.embed(query_locs).view(B, -1, self.embedder.out_dim)
            query_locs = query_locs.transpose(1, 2).contiguous() # TODO check shape [B, 2, N]

        pix_feature = geom_featmap # [B, C, N]

        if self.incl_query_nml:
            query_locs = torch.cat([query_locs, query_loc_nml], 1) # [B, C, N]

        w0 = 30.0 if self.actv_fn == 'sin' else 1.0
        net_input = torch.cat([pix_feature, query_locs], 1)

        '''
        decoder
        '''
        pred_res, pred_normals = self.decoder(net_input, local_feat_gain=w0)

        if non_handfeet_mask is not None:
            pred_res = pred_res * non_handfeet_mask.unsqueeze(0).unsqueeze(0)

        pred_res_cano = pred_res.clone().permute(0,2,1)

        '''
        local to global transform
        '''
        pred_clo_cano = cano_pt_minimal + pred_res * transf_scaling
        pred_clo_cano_mean = query_loc_3d + pred_res * transf_scaling # add to the canonical body without pose blendshapes
        
        pred_normals_cano = pred_normals.transpose(2,1)
        pred_normals_cano = F.normalize(pred_normals_cano, dim=-1)

        pred_T = torch.einsum('bjst,bjv->bvst', jT, smpl_lbsw_local) 
        pred_T[:,:,3,3] = 1.0

        pred_res = torch.einsum('bvst,btv->bvs', pred_T, homogenize(pred_res, 1, is_direction=True))[:,:,:3]
        pred_res = pred_res * transf_scaling
        pred_body_posed = torch.einsum('bvst,btv->bvs', pred_T, homogenize(cano_pt_minimal, 1, is_direction=False))[:,:,:3]
        pred_body_nml_posed = torch.einsum('bvst,btv->bvs', pred_T, homogenize(query_loc_nml, 1, is_direction=True))[:,:,:3]

        if self.transf_only_disp:
            pred_clo_posed = pred_res + posed_pt_minimal.transpose(2,1)
        else:
            pred_clo_posed = pred_res + pred_body_posed

        pred_normals = torch.einsum('bvst,btv->bvs', pred_T, homogenize(pred_normals, 1, is_direction=True))[:,:,:3]
        pred_normals = F.normalize(pred_normals, dim=-1)

        if non_handfeet_mask is not None:
            hand_feet_pt_idx = (non_handfeet_mask !=1)
            pred_normals[:, hand_feet_pt_idx, :] = pred_body_nml_posed[:, hand_feet_pt_idx, :]

        preds = {
            'normals_posed': pred_normals,
            'normals_cano': pred_normals_cano,
            'disps_posed': pred_res,
            'disps_cano': pred_res_cano,
            'clothed_posed': pred_clo_posed,
            'clothed_cano': pred_clo_cano.transpose(2,1),
            'clothed_cano_mean': pred_clo_cano_mean.transpose(2,1),
            'pred_body_posed': pred_body_posed,
            'pred_body_nml_posed': pred_body_nml_posed,
            'gt_body_posed': posed_pt_minimal.transpose(2,1),
            'pred_lbsw': smpl_lbsw_local,
            'gt_lbsw': smpl_lbsw_local,
        } # all have shape [B, num_pts, 3], lbsw has shape [B, num_joints, num_pts]
        return preds
