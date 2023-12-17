import torch
torch.backends.cudnn.deterministic = True

from lib.losses import normal_loss, chamfer_loss_separate
from lib.utils_model import interp_features_dev, pix_coord_convert, flip_pix_coords_up_down, normalize_uv
from lib.utils_pose import batch_rod2quat
from lib.utils_train import adaptive_sampling, sample_points_from_meshes, get_adaptive_point_loss_weights

import numpy as np
from os.path import join, dirname, realpath

from pytorch3d.structures import Meshes
import tqdm

SCRIPT_DIR = dirname(realpath(__file__))


def train(
        model, geom_featmap, train_loader, optimizer,
        device='cuda',
        loss_weights=None,
        transf_scaling=1.0,
        hires_assets=None,
        use_hires_smpl=False,
        uv_coord_map=None, 
        valid_idx=None,
        train_progress=0.0, 
        **kwargs,
        ):
    
    is_hires = '_hires' if hires_assets is not None else ''
    body_model_type = kwargs['body_model_type']
    
    n_train_samples = len(train_loader.dataset)

    train_s2m, train_m2s, train_lnormal, train_lbsw_loss, train_reproj_loss, train_rgl, train_latent_rgl, train_corr_rgl, train_total = 0, 0, 0, 0, 0, 0, 0, 0, 0
    w_s2m, w_m2s, w_normal, w_lbsw, w_reproj, w_rgl, w_latent_rgl, w_corr_rgl = loss_weights

    model.train()
    query_bary, query_face_idx, query_uv = [], [], []

    # regular grid points
    query_posmap_size = kwargs['query_posmap_size']
    query_bary.append(torch.tensor(np.load(join(SCRIPT_DIR, '..', 'assets', 'barycoords_{}{}_pop_grid_{}.npy'.format(body_model_type, is_hires, query_posmap_size) ))).to(device).unsqueeze(0))
    query_face_idx.append(torch.tensor(np.load(join(SCRIPT_DIR, '..', 'assets', 'faceidx_{}{}_pop_grid_{}.npy'.format(body_model_type, is_hires, query_posmap_size) ))).to(device).unsqueeze(0).long())
    grid_uv_coords = interp_features_dev(faces=model.faces_uv, vert_features=model.verts_uv[None, :].transpose(1,2), bary_coords=query_bary[0], sample_face_idxs=query_face_idx[0].squeeze(0))
    grid_uv_coords = flip_pix_coords_up_down(grid_uv_coords)

    if kwargs['align_corners']:
        grid_uv_coords = (pix_coord_convert(grid_uv_coords) * 255).round() / 255.
    if kwargs['use_original_grid']: # use the original positional map grid as in the POP paper
        grid_uv_coords = uv_coord_map[valid_idx].unsqueeze(0)
    query_uv.append(grid_uv_coords)

    # random sample points on body surface prop. to tri area
    if kwargs['num_pt_random_train'] > 0:
        mesh_to_sample = Meshes(verts=model.cano_vt[None,:], faces=model.faces[None,:])
        query_locs_rand, query_face_idx_rand, query_bary_rand = sample_points_from_meshes(meshes=mesh_to_sample, num_samples=kwargs['num_pt_random_train'], return_bary_coords=True)

        # get uv image-plane pixel coords for the sampled points
        # note: these uv coords treats each pixel as a square and its coord is the center of the square (corresponds to align_corners=False in F.gridsample)
        rand_uv_coords = interp_features_dev(faces=model.faces_uv, vert_features=model.verts_uv[None, :].transpose(1,2), bary_coords=query_bary_rand, sample_face_idxs=query_face_idx_rand.squeeze(0))
        rand_uv_coords = flip_pix_coords_up_down(rand_uv_coords)

        if (kwargs['align_corners'] or kwargs['use_original_grid']):
            rand_uv_coords = (pix_coord_convert(rand_uv_coords) * 255).round() / 255.
        
        query_bary.append(query_bary_rand)
        query_face_idx.append(query_face_idx_rand)
        query_uv.append(rand_uv_coords)

    query_bary, query_face_idx, query_uv = list(map(lambda x: torch.cat(x, dim=1), [query_bary, query_face_idx, query_uv]))
    query_uv = query_uv.unsqueeze(2) # used for F.grid_sample later

    for iterno, data in enumerate(tqdm.tqdm(train_loader)):
        # -------------------------------------------------------
        # ------------ load batch data and reshaping ------------
        [query_posmap, pose_inp_tensor, target_pc_n, target_pc, vtransf, jT, target_names, body_pose, posed_subj_v, posed_mean_v, cano_subj_bs_v, index] = data 

        gpu_data =  [query_posmap, pose_inp_tensor, target_pc_n, target_pc, vtransf, jT, body_pose, posed_subj_v, posed_mean_v, cano_subj_bs_v, index]
        [query_posmap, pose_inp_tensor, target_pc_n, target_pc, vtransf, jT, body_pose, posed_subj_v, posed_mean_v, cano_subj_bs_v, index] = list(map(lambda x: x.to(device), gpu_data))
        bs, _, H, W = query_posmap.size()

        optimizer.zero_grad()

        # original geom feat map: [num_outfits, C, H, W]
        # each clotype (the 'index' when loading the data) uses a unique [C, H, W] slice for all its frames
        geom_featmap_batch = geom_featmap[index, ...]

        body_pose = batch_rod2quat(body_pose.reshape(-1, 3)).view(bs, -1, 4)

        query_uv_batch = query_uv.expand(bs, -1, -1, -1).contiguous()

        if use_hires_smpl:
            posed_subj_v_extra = interp_features_dev(hires_assets['face_original'], posed_subj_v.permute(0,2,1), hires_assets['bary'].expand(bs,-1,-1), hires_assets['faceid'])
            posed_subj_v = torch.cat([posed_subj_v, posed_subj_v_extra], dim=1) # body verts in resynth data are of original smpl/smplx resolution; upsample them here

            cano_subj_bs_v_extra = interp_features_dev(hires_assets['face_original'], cano_subj_bs_v.permute(0,2,1), hires_assets['bary'].expand(bs,-1,-1), hires_assets['faceid'])
            cano_subj_bs_v = torch.cat([cano_subj_bs_v, cano_subj_bs_v_extra], dim=1) # body verts in resynth data are of original smpl/smplx resolution; upsample them here


        # --------------------------------------------------------------------
        # ------------ model pass an coordinate transformation ---------------

        if (kwargs['coarse_shapes'] is not None):
            coarse_clo_pts = kwargs['coarse_shapes'][index]
            coarse_clo_pts = coarse_clo_pts.permute(0,2,1).contiguous()
        else:
            coarse_clo_pts = None
        
        if (kwargs['diffused_lbsws'] is not None):
            pre_diffused_lbsw = kwargs['diffused_lbsws'][index]
            pre_diffused_lbsw = pre_diffused_lbsw.permute(0,2,1).contiguous()
        else:
            pre_diffused_lbsw = None

        if kwargs['non_handfeet_mask'] is not None:
            non_handfeet_mask = kwargs['non_handfeet_mask'].bool()
        else:
            non_handfeet_mask = None
            
        preds = model(  
                        pose_inp_tensor, 
                        geom_featmap=geom_featmap_batch,
                        query_uv=query_uv_batch,
                        query_bary=query_bary,
                        query_face_idx=query_face_idx,
                        pose_params=body_pose,
                        jT=jT, 
                        posed_subj_v=posed_subj_v,
                        cano_subj_bs_v=cano_subj_bs_v,
                        coarse_clo_pts=coarse_clo_pts,
                        pre_diffused_lbsw=pre_diffused_lbsw,
                        transf_scaling=transf_scaling,
                        interp_same_bary_idx=True,
                        align_corners=kwargs['align_corners'],
                        train_progress=train_progress,
                        non_handfeet_mask=non_handfeet_mask if query_posmap_size==256 else None,
                    )

        pred_res, pred_res_cano, pred_normals = preds['disps_posed'], preds['disps_cano'], preds['normals_posed']
        pred_lbsw, gt_lbsw, pred_pts_posed, pred_pts_cano = preds['pred_lbsw'], preds['gt_lbsw'], preds['clothed_posed'], preds['clothed_cano']
        pred_minimal_posed, gt_minimal_posed = preds['pred_body_posed'], preds['gt_body_posed']

        if kwargs['adaptive_sample_in_training']:
            sampled_pts_cano, sampled_pts_posed, sampled_face_idx_new, sampled_bary_new = adaptive_sampling(pred_pts_posed, model, query_face_idx, max_power=7,
                                                                                                posed_subj_v=posed_subj_v, num_pt_adaptive=4000)
            adaptive_uv = interp_features_dev(faces=model.faces_uv.expand(bs, -1, -1), vert_features=model.verts_uv.transpose(0,1).expand(bs, -1, -1),
                                            bary_coords=sampled_bary_new, sample_face_idxs=sampled_face_idx_new, interp_same_bary_idx=False)
            adaptive_uv = flip_pix_coords_up_down(adaptive_uv)
            adaptive_uv = adaptive_uv.unsqueeze(2)

            if (kwargs['align_corners'] or kwargs['use_original_grid']):
                adaptive_uv = (pix_coord_convert(adaptive_uv) * 255).round() / 255.
            
            preds_new = model(
                            pose_inp_tensor, # mean shape, posed body positional maps as input to the network
                            geom_featmap=geom_featmap_batch,
                            query_uv=adaptive_uv,
                            query_bary=sampled_bary_new,
                            query_face_idx=sampled_face_idx_new,
                            jT=jT,
                            pose_params=body_pose,
                            transf_scaling=transf_scaling,
                            posed_subj_v=posed_subj_v,
                            cano_subj_bs_v=cano_subj_bs_v,
                            interp_same_bary_idx=False,
                            align_corners=kwargs['align_corners']
                            )
                            
            pred_normals2, pred_pts_posed2 = preds_new['normals_posed'], preds_new['clothed_posed']

            pred_pts_posed = torch.cat([pred_pts_posed, pred_pts_posed2], 1)
            pred_normals = torch.cat([pred_normals, pred_normals2], 1)

        # --------------------------------
        # ------------ losses ------------
        # Chamfer dist from the (s)can to (m)odel: from the GT points to its closest ponit in the predicted point set
        m2s, s2m, idx_closest_gt, _ = chamfer_loss_separate(pred_pts_posed, target_pc) #idx1: [#pred points]
        s2m = torch.mean(s2m)
        
        # normal loss
        lnormal, closest_target_normals = normal_loss(pred_normals, target_pc_n, idx_closest_gt)

        # dist from the predicted points to their respective closest point on the GT, projected by
        # the normal of these GT points, to appxoimate the point-to-surface distance
        nearest_idx = idx_closest_gt.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
        target_points_chosen = torch.gather(target_pc, dim=1, index=nearest_idx)
        pc_diff = target_points_chosen - pred_pts_posed # vectors from prediction to its closest point in gt pcl
        m2s = torch.sum(pc_diff * closest_target_normals, dim=-1) # project on direction of the normal of these gt points
        m2s = torch.mean(m2s**2) # the length (squared) is the approx. pred point to scan surface dist.

        if ((kwargs['adaptive_weight_in_training']) or (kwargs['adaptive_lbsw_weight_in_training'])):
            adapt_pt_baseweights = get_adaptive_point_loss_weights(pred_pts_posed, base_weight=torch.tensor(1.0), max_power=10, use_thres=True, binary=True)

        if kwargs['adaptive_weight_in_training']:
            w_rgl_adapt = adapt_pt_baseweights.unsqueeze(-1) * w_rgl
            rgl_len = torch.mean((pred_res ** 2)*w_rgl_adapt)
        else:
            rgl_len = torch.mean((pred_res ** 2)*w_rgl)

        if kwargs['adaptive_lbsw_weight_in_training']:
            w_reproj_adapt = adapt_pt_baseweights * w_reproj
            w_lbsw_adapt = adapt_pt_baseweights * w_lbsw
            w_lbsw_adapt = w_lbsw_adapt.unsqueeze(1) # to match the [B, n_joints, n_pts] shape of lbsw loss below

            lbsw_loss = torch.mean(torch.nn.L1Loss(reduction='none')(pred_lbsw, gt_lbsw) * w_lbsw_adapt)
            reproj_loss = torch.mean((pred_minimal_posed - gt_minimal_posed).pow(2).sum(-1) * w_reproj_adapt)
        else:
            lbsw_loss = torch.nn.L1Loss()(pred_lbsw, gt_lbsw) * w_lbsw
            reproj_loss = (pred_minimal_posed - gt_minimal_posed).pow(2).sum(-1).mean() * w_reproj

        rgl_latent = torch.mean(geom_featmap_batch**2)
        
        # regulrization term for a consistent correspondence (but not too strong that the disps are pose-independent)
        if kwargs['use_variance_rgl']:
            if bs < 3:
                rgl_corr = torch.tensor(0.) # variance can't be calculated across batch if there's only 1 example
            else:
                rgl_corr = torch.std(pred_res_cano, dim=0).mean()
        else:
            rgl_corr = torch.tensor(0.)

        loss = s2m*w_s2m + m2s*w_m2s + lnormal* w_normal + rgl_len + rgl_latent*w_latent_rgl + lbsw_loss  + reproj_loss + rgl_corr * w_corr_rgl

        loss.backward()
        optimizer.step()

        # ------------------------------------------
        # ------------ accumulate stats ------------

        train_m2s += m2s * bs
        train_s2m += s2m * bs
        train_lnormal += lnormal * bs
        train_lbsw_loss += lbsw_loss * bs
        train_reproj_loss += reproj_loss * bs
        train_rgl += rgl_len * bs
        train_latent_rgl += rgl_latent * bs
        train_corr_rgl += rgl_corr * bs

        train_total += loss * bs

    train_s2m /= n_train_samples
    train_m2s /= n_train_samples
    train_lnormal /= n_train_samples
    train_lbsw_loss /= n_train_samples
    train_reproj_loss /= n_train_samples
    train_rgl /= n_train_samples
    train_latent_rgl /= n_train_samples
    train_corr_rgl /= n_train_samples
    train_total /= n_train_samples

    return train_m2s, train_s2m, train_lnormal, train_lbsw_loss, train_reproj_loss, train_rgl, train_latent_rgl, train_corr_rgl, train_total

