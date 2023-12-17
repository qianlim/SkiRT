from distutils.cygwinccompiler import Mingw32CCompiler
from os.path import join
from re import I
import torch
torch.backends.cudnn.deterministic = True
from tqdm import tqdm

from lib.utils_io import save_result_examples
from lib.losses import normal_loss, chamfer_loss_separate, uniformity_metric
from lib.utils_model import interp_features_dev, pix_coord_convert, flip_pix_coords_up_down
from lib.utils_pose import batch_rod2quat
from lib.utils_train import adaptive_sampling, unpack_face_idx_by_batch


import numpy as np
from os.path import join, dirname, realpath

SCRIPT_DIR = dirname(realpath(__file__))

def test_seen_clo(
                model,
                geom_featmap, 
                test_loader, 
                epoch_idx, 
                samples_dir, 
                uv_coord_map=None, 
                valid_idx=None,
                model_name=None, 
                transf_scaling=1.0,
                save_all_results=False,
                device='cuda',
                mode='val', # val, test_seen
                hires_assets=None,
                use_hires_smpl=False,
                adaptive_sample_loops=3,
                **kwargs,
    ):
    '''
    If the test outfit is seen, just use the optimal clothing code found during training
    '''
    model.eval()
    print('Evaluating...')

    n_test_samples = len(test_loader.dataset)

    test_s2m, test_m2s, test_lnormal, test_lbsw_loss, test_reproj_loss, test_rgl, test_latent_rgl, test_corr_rgl = 0, 0, 0, 0, 0, 0, 0, 0

    is_hires = '_hires' if hires_assets is not None else ''
    body_model_type = kwargs['body_model_type']
    
    with torch.no_grad():
        for data in tqdm(test_loader):
            query_bary, query_face_idx, query_uv = [], [], []

            if (kwargs['eval_body_verts'] and (kwargs['vert_bary'] is not None)):
                query_bary.append(kwargs['vert_bary'])
                query_face_idx.append(kwargs['vert_fid'])
            else:
                # use only regular grid points for testing
                query_posmap_size = kwargs['query_posmap_size']
                query_bary.append(torch.tensor(np.load(join(SCRIPT_DIR, '..', 'assets', 'barycoords_{}{}_pop_grid_{}.npy'.format(body_model_type, is_hires, query_posmap_size) ))).to(device).unsqueeze(0))
                query_face_idx.append(torch.tensor(np.load(join(SCRIPT_DIR, '..', 'assets', 'faceidx_{}{}_pop_grid_{}.npy'.format(body_model_type, is_hires, query_posmap_size) ))).to(device).unsqueeze(0).long())
            grid_uv_coords = interp_features_dev(faces=model.faces_uv, vert_features=model.verts_uv[None, :].transpose(1,2), bary_coords=query_bary[0], sample_face_idxs=query_face_idx[0].squeeze(0))
            grid_uv_coords = flip_pix_coords_up_down(grid_uv_coords)
            
            if kwargs['align_corners']:
                grid_uv_coords = (pix_coord_convert(grid_uv_coords) * 255).round() / 255.
            if kwargs['use_original_grid']:
                grid_uv_coords = uv_coord_map[valid_idx].unsqueeze(0)
            
            query_uv.append(grid_uv_coords)
            query_bary, query_face_idx, query_uv = list(map(lambda x: torch.cat(x, dim=1), [query_bary, query_face_idx, query_uv]))
            query_uv = query_uv.unsqueeze(2)

            if kwargs['non_handfeet_mask'] is not None:
                non_handfeet_mask = kwargs['non_handfeet_mask'].bool()
            else:
                non_handfeet_mask = None

            # -------------------------------------------------------
            # ------------ load batch data and reshaping ------------
            
            [query_posmap, pose_inp_tensor, target_pc_n, target_pc, vtransf, jT, target_names, body_pose, posed_subj_v, posed_mean_v, cano_subj_bs_v, index] = data 
            gpu_data =  [query_posmap, pose_inp_tensor, target_pc_n, target_pc, vtransf, jT, body_pose, posed_subj_v, posed_mean_v, cano_subj_bs_v, index]
            [query_posmap, pose_inp_tensor, target_pc_n, target_pc, vtransf, jT, body_pose, posed_subj_v, posed_mean_v, cano_subj_bs_v, index] = list(map(lambda x: x.to(device, non_blocking=True), gpu_data))

            bs, _, H, W = query_posmap.size()
            
            geom_featmap_batch = geom_featmap[index, ...]

            body_pose = batch_rod2quat(body_pose.reshape(-1, 3)).view(bs, -1, 4)

            query_uv_batch = query_uv.expand(bs, -1, -1, -1).contiguous()

            if use_hires_smpl:
                posed_subj_v_extra = interp_features_dev(hires_assets['face_original'], posed_subj_v.permute(0,2,1), hires_assets['bary'].expand(bs,-1,-1), hires_assets['faceid'])
                posed_subj_v = torch.cat([posed_subj_v, posed_subj_v_extra], dim=1) # body verts in resynth data are of original smpl/smplx resolution; upsample them here

                cano_subj_bs_v_extra = interp_features_dev(hires_assets['face_original'], cano_subj_bs_v.permute(0,2,1), hires_assets['bary'].expand(bs,-1,-1), hires_assets['faceid'])
                cano_subj_bs_v = torch.cat([cano_subj_bs_v, cano_subj_bs_v_extra], dim=1) # body verts in resynth data are of original smpl/smplx resolution; upsample them here

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

            # --------------------------------------------------------------------------------------------
            # ------------ model forward pass and coordinate transformation of predictions ---------------
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
                            non_handfeet_mask=kwargs['non_handfeet_mask'] if ('test' in mode) else None,
                        )

            pred_res, pred_res_cano, pred_normals, pred_normals_cano = preds['disps_posed'], preds['disps_cano'], preds['normals_posed'], preds['normals_cano']
            pred_lbsw, gt_lbsw, pred_pts_posed, pred_pts_cano = preds['pred_lbsw'], preds['gt_lbsw'], preds['clothed_posed'], preds['clothed_cano']
            pred_minimal_posed, gt_minimal_posed = preds['pred_body_posed'], preds['gt_body_posed']

            if 'clothed_cano_mean' in preds.keys():
                pred_pts_cano_mean = preds['clothed_cano_mean']

            pred_pts_posed_orig = pred_pts_posed.clone()
            pred_normals_orig = pred_normals.clone()
            pred_normals_cano_orig = pred_normals_cano.clone()
            pred_pts_cano_orig = pred_pts_cano.clone()
            pred_lbsw_orig = pred_lbsw.clone() # for coarse stage, lbsw isn't predicted, so returned the body model's lbsw

            '''adaptive sampling'''
            if adaptive_sample_loops != 0:
                query_face_idx_non_handfeet = query_face_idx[:, non_handfeet_mask]
                pred_pts_posed_non_handfeet = pred_pts_posed[:, non_handfeet_mask]
                initial_pred_pts_non_handfeet = pred_pts_posed_non_handfeet.shape[1]
            else:
                pred_pts_posed_non_handfeet = pred_pts_posed
                initial_pred_pts_non_handfeet = pred_pts_posed_non_handfeet.shape[1]

            for i in range(adaptive_sample_loops):
                
                # sampled_pts_cano, sampled_pts_posed, sampled_face_idx_new, sampled_bary_new = adaptive_sampling(pred_pts_posed, model, query_face_idx, posed_subj_v=posed_subj_v, num_pt_adaptive=kwargs['num_pt_adaptve'] // (i+1))
                sampled_pts_cano, sampled_pts_posed, sampled_face_idx_new, sampled_bary_new = \
                            adaptive_sampling(pred_pts_posed_non_handfeet, model, query_face_idx_non_handfeet, posed_subj_v=posed_subj_v, num_pt_adaptive=kwargs['num_pt_adaptve'] // (i+1))

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
                                align_corners=kwargs['align_corners'],
                                non_handfeet_mask=None, # the adaptive sampled points already exclude handfeet, no need for mask here
                                )

                pred_normals2, pred_normals_cano2, pred_pts_posed2 = preds_new['normals_posed'], preds_new['normals_cano'], preds_new['clothed_posed']
                pred_lbsw2, gt_lbsw2, pred_pts_cano2 = preds_new['pred_lbsw'], preds_new['gt_lbsw'], preds_new['clothed_cano']
                pred_minimal_posed2, gt_minimal_posed2 = preds_new['pred_body_posed'], preds_new['gt_body_posed'], 


                pred_pts_posed_non_handfeet = torch.cat([pred_pts_posed_non_handfeet, pred_pts_posed2], 1)
                pred_normals = torch.cat([pred_normals, pred_normals2], 1)
                pred_normals_cano = torch.cat([pred_normals_cano, pred_normals_cano2], 1)
                pred_lbsw = torch.cat([pred_lbsw, pred_lbsw2], -1)
                gt_lbsw = torch.cat([gt_lbsw, gt_lbsw2], -1)
                pred_pts_cano = torch.cat([pred_pts_cano, pred_pts_cano2], 1)
                pred_minimal_posed = torch.cat([pred_minimal_posed, pred_minimal_posed2], 1)
                gt_minimal_posed = torch.cat([gt_minimal_posed, gt_minimal_posed2], 1)

                sampled_face_idx_new = unpack_face_idx_by_batch(sampled_face_idx_new, len(model.faces))
                if len(query_face_idx_non_handfeet) == 1:
                    query_face_idx_non_handfeet = query_face_idx_non_handfeet.expand(bs, -1)
                total_sampled_face_idx_now = torch.cat([query_face_idx_non_handfeet, sampled_face_idx_new], dim=-1)
                query_face_idx_non_handfeet = total_sampled_face_idx_now

            # add back hand and feet preds (disps=0) from the initial round prediction using regular grid as query points
            pred_pts_posed = torch.cat([pred_pts_posed, pred_pts_posed_non_handfeet[:, initial_pred_pts_non_handfeet:]], 1)

            # --------------------------------
            # ------------ losses ------------

            _, s2m, idx_closest_gt, _ = chamfer_loss_separate(pred_pts_posed, target_pc) #idx1: [#pred points]
            s2m = s2m.mean(1)
            lnormal, closest_target_normals = normal_loss(pred_normals, target_pc_n, idx_closest_gt, phase='test')
            nearest_idx = idx_closest_gt.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
            target_points_chosen = torch.gather(target_pc, dim=1, index=nearest_idx)
            pc_diff = target_points_chosen - pred_pts_posed # vectors from prediction to its closest point in gt pcl
            m2s = torch.sum(pc_diff * closest_target_normals, dim=-1) # project on direction of the normal of these gt points
            m2s = torch.mean(m2s**2, 1) # the length (squared) is the approx. pred point to scan surface dist.

            rgl_len = torch.mean((pred_res ** 2).reshape(bs, -1),1)
            rgl_latent = torch.mean(geom_featmap_batch**2)

            lbsw_loss = torch.nn.L1Loss()(pred_lbsw, gt_lbsw)
            reproj_loss = torch.nn.MSELoss()(pred_minimal_posed, gt_minimal_posed)

            if bs < 3:
                corr_rgl = torch.tensor(0.)
            else:
                corr_rgl = torch.std(pred_res_cano, dim=0).mean()

            # knn_rad_mean, knn_rad_std, knn_rad_med, knn_rad_max = uniformity_metric(pred_pts_posed)

            # ------------------------------------------
            # ------------ accumulate stats ------------

            test_m2s += torch.sum(m2s)
            test_s2m += torch.sum(s2m)
            test_lnormal += torch.sum(lnormal)
            test_lbsw_loss += torch.sum(lbsw_loss)
            test_reproj_loss += torch.sum(reproj_loss)
            test_rgl += torch.sum(rgl_len)
            test_latent_rgl += rgl_latent
            test_corr_rgl += corr_rgl
            
            if 'test' in mode:
                save_spacing = 1 if save_all_results else 10
                if len(pred_pts_posed) == 1:
                    save_spacing = 1

                vert_connectivity = model.faces.cpu().numpy() if kwargs['eval_body_verts'] else None

                for i in range(pred_pts_posed.shape[0])[::save_spacing]:
                    lbsw_to_save_orig = pred_lbsw_orig[0].transpose(1,0) if kwargs['save_lbsw'] else None

                    save_result_examples(samples_dir, model_name, target_names[i],
                                        points=pred_pts_posed_orig[i], normals=pred_normals_orig[i], lbsw=None, vert_connectivity=vert_connectivity)
                    save_result_examples(samples_dir, model_name, target_names[i],
                                        points=pred_pts_cano_orig[i], normals=pred_normals_cano_orig[i], lbsw=lbsw_to_save_orig, is_cano=True, vert_connectivity=vert_connectivity)
                    save_result_examples(samples_dir, model_name, target_names[i],
                                        points=pred_pts_posed_orig[i], normals=None, lbsw=lbsw_to_save_orig, vert_connectivity=vert_connectivity)

                    # for coarse stage prediction, need to output cano clothing on mean smpl body (for querying the pre-diffused lbsw)
                    # and the personal body (for the fine stage-add-on)
                    if 'clothed_cano_mean' in preds.keys(): 
                        save_result_examples(samples_dir, model_name, target_names[i],
                                            points=pred_pts_cano_mean[i], normals=pred_normals_cano[i], extra_suffix='_cano_mean')

                    if adaptive_sample_loops > 0:
                        save_result_examples(samples_dir, model_name, target_names[i]+'_adaptive{}'.format(adaptive_sample_loops),
                                            points=pred_pts_posed[i], normals=pred_normals[i], lbsw=None)


    test_m2s /= n_test_samples
    test_s2m /= n_test_samples
    test_lnormal /= n_test_samples
    test_lbsw_loss /= n_test_samples
    test_reproj_loss /= n_test_samples
    test_rgl /= n_test_samples
    test_latent_rgl /= n_test_samples
    test_corr_rgl /= n_test_samples

    test_s2m, test_m2s, test_lnormal, test_lbsw_loss, test_reproj_loss, test_rgl, test_latent_rgl, test_corr_rgl = list(map(lambda x: x.detach().cpu().numpy(), [test_s2m, test_m2s, test_lnormal, test_lbsw_loss, test_reproj_loss, test_rgl, test_latent_rgl, test_corr_rgl]))

    print("model2scan dist: {:.3e}, scan2model dist: {:.3e}, normal loss: {:.3e}, lbsw loss: {:.3e}, reproj loss: {:.3e}"
          " rgl term: {:.3e}, latent rgl term:{:.3e}, corresp rgl term: {:.3e}".format(test_m2s, test_s2m, test_lnormal, test_lbsw_loss, test_reproj_loss,
                                                              test_rgl, test_latent_rgl, test_corr_rgl))

    # for validation, save one example at every X epochs for inspection
    if mode == 'val':
        if epoch_idx == 0 or epoch_idx % 20 == 0:
            lbsw_to_save = pred_lbsw[0].transpose(1,0) if kwargs['save_lbsw'] else None
            save_result_examples(samples_dir, model_name, target_names[0],
                                points=pred_pts_posed[0], normals=pred_normals[0], lbsw=lbsw_to_save,
                                patch_color=None, epoch=epoch_idx)
        
    return [test_s2m, test_m2s, test_lnormal, test_lbsw_loss, test_reproj_loss, test_rgl, test_latent_rgl, test_corr_rgl]