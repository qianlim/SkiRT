import os
from os.path import join, basename, dirname, realpath
import sys
import time
from datetime import datetime
from psbody.mesh import Mesh

PROJECT_DIR = dirname(realpath(__file__))
sys.path.append(PROJECT_DIR)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
torch.backends.cudnn.deterministic = True


from lib.config_parser import parse_config, load_net_config
from lib.dataset import CloDataSet
from lib.network import SkiRT_fine, SkiRT_Coarse
from lib.train import train
from lib.infer import test_seen_clo
from lib.utils_io import save_model, save_latent_feats, load_latent_feats
from lib.utils_train import adjust_loss_weights
from lib_data.data_paths import DataPaths
from lib.load_assets import BodyModelAssets

torch.manual_seed(12345)
np.random.seed(12345)

DEVICE = torch.device('cuda')

def main():
    args = parse_config()

    dpth = DataPaths()
    dpth.set_up_experiment_paths(args, project_dir=PROJECT_DIR)

    clo_body_assets = BodyModelAssets(args, dpth, DEVICE)

    exp_name = args.name

    lbs_net_opt = load_net_config(args.model_config)['lbs_net_scanimate']
    lbs_net_opt['body_model_type'] = clo_body_assets.body_model_type
    shape_mlp_opt = load_net_config(args.model_config)['shape_decoder']
    
    archi_args = {
                    'input_nc':3,
                    'num_emb_freqs': args.num_emb_freqs,
                    'c_geom': args.c_geom,
                    'inp_posmap_size': args.inp_posmap_size,
                    'hsize': args.hsize,
                    'pos_encoding': bool(args.pos_encoding),
                    'num_emb_freqs': args.num_emb_freqs,
                    'posemb_incl_input': bool(args.posemb_incl_input),
                    'query_coord_dim': 3 if bool(args.query_xyz) else 2,

                    # some smpl-related assets
                    'cano_query_pts': clo_body_assets.query_locs_3d,
                    'smpl_cano_vt': clo_body_assets.cano_v.astype(float),
                    'smpl_cano_vnormal': clo_body_assets.cano_vn.astype(float),
                    'smpl_f': clo_body_assets.cano_f.astype(np.int32),
                    'smpl_v_uv': clo_body_assets.cano_vt.astype(float),
                    'smpl_f_uv': clo_body_assets.cano_ft.astype(np.int32),
                    'body_lbsw': clo_body_assets.vert_lbsw,

                    # some training options
                    'incl_query_nml': bool(args.incl_query_nml),
                    'query_xyz': bool(args.query_xyz),
                    'use_vert_geom_feat': bool(args.use_vert_geom_feat),
                    'use_global_geom_feat': bool(args.use_global_geom_feat),
                    'use_pose_emb': bool(args.use_pose_emb),
                    'use_jT': bool(args.use_jT),
                    'use_pred_lbsw': bool(args.pred_lbsw),
                    'transf_only_disp': bool(args.transf_only_disp),
                    'lbs_net_opt': lbs_net_opt,
                    'shape_mlp_opt': shape_mlp_opt,
                }

    # build_model
    if args.stage == 'coarse':
        model = SkiRT_Coarse(**archi_args)
    else:
        archi_args_fine = {
            'pose_feat_type': args.pose_feat_type.lower(),
            'pose_input': args.pose_input.lower(),
            'pose_map': clo_body_assets.pose_map,
            'gradual_pred_lbsw': False,
            'c_pose': args.c_pose, # channels for pose features
            'nf': args.nf, # number of filters in network
            'up_mode': args.up_mode,
            'use_dropout': bool(args.use_dropout),
        }
        archi_args.update(archi_args_fine)
        model = SkiRT_fine(**archi_args)
    print(model)

    ## build the optimizable geometric feature map
    if bool(args.use_global_geom_feat):# a single global vector for geometric features (used for coarse xtage)
        geom_featmap = torch.ones(clo_body_assets.num_outfits_seen, args.c_geom, 1).normal_(mean=0., std=0.01).cuda()
    else:
        if bool(args.use_vert_geom_feat): # local geometric features at each vertex
            num_verts = len(clo_body_assets.cano_v)
            geom_featmap = torch.ones(clo_body_assets.num_outfits_seen, args.c_geom, num_verts).normal_(mean=0., std=0.01).cuda()
        else: #  local geometric features at each pixel (a point on the body) of the uv positional map (used for fine stage)
            geom_featmap = torch.ones(clo_body_assets.num_outfits_seen, args.c_geom, args.inp_posmap_size, args.inp_posmap_size).normal_(mean=0., std=0.01).cuda()
    geom_featmap.requires_grad = True
    print(geom_featmap.shape)


    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": geom_featmap, "lr": args.lr_geomfeat}
        ])

    n_epochs = args.epochs
    epoch_now = 0

    dataset_args = {
                     'dataset_type': args.dataset_type,
                     'body_model':  clo_body_assets.body_model_type,
                     'data_root': dpth.data_root,
                     'data_root_extra': dpth.data_root_extra, 
                     'scan_root': dpth.scan_root,
                     'use_raw_scan': bool(args.use_raw_scan),
                     'query_posmap_size':args.query_posmap_size, # query positional map resolution
                     'inp_posmap_size': args.inp_posmap_size, # (model) input positional map (as pose information) resolution
                     'pose_input': args.pose_input,
                     }
    
    training_args = {
                        'body_model_type':  clo_body_assets.body_model_type,
                        'device': DEVICE,
                        'flist_uv': clo_body_assets.flist_uv,
                        'valid_idx':  clo_body_assets.valid_idx,
                        'uv_coord_map':  clo_body_assets.uv_coord_map,
                        'query_posmap_size': args.query_posmap_size,
                        'cano_query_pts': clo_body_assets.query_locs_3d,
                        'cano_query_nml': clo_body_assets.query_nml,
                        'bary_coords_map': clo_body_assets.bary_coords,
                        'transf_scaling': args.transf_scaling,
                        'query_xyz': bool(args.query_xyz),
                        'hires_assets': clo_body_assets.hires_assets,
                        "use_hires_smpl": bool(args.use_hires_smpl),
                        'num_pt_adaptve': args.num_pt_adaptve,
                        'save_lbsw': False,
                        'adaptive_sample_loops': args.adaptive_sample_loops,
                        'adaptive_sample_in_training': bool(args.adaptive_sample_in_training), # will be set to True after X epochs (set during training)
                        'adaptive_weight_in_training': bool(args.adaptive_weight_in_training),
                        'adaptive_lbsw_weight_in_training': bool(args.adaptive_lbsw_weight_in_training), # will be set to True after X epochs (set during training)
                        'num_pt_random_train': args.num_pt_random_train,
                        'use_original_grid': bool(args.use_original_grid),
                        'align_corners': True,
                        'coarse_shapes': None, 
                        'diffused_lbsws': None,
                        'vert_bary': clo_body_assets.vert_bary,
                        'vert_fid': clo_body_assets.vert_fid, #fid： face id
                        'eval_body_verts': bool(args.eval_body_verts),
                        'use_variance_rgl': bool(args.use_variance_rgl),
                        'single_direc_chamfer': bool(args.single_direc_chamfer),
                        'non_handfeet_mask': clo_body_assets.non_handfeet_mask if bool(args.exclude_handfeet) else None,
                    }

    '''
    ------------ Load checkpoints in case of test or resume training ------------
    '''
    if args.mode.lower() in ['resume', 'test', 'test_seen']:
        checkpoints = sorted([fn for fn in os.listdir(dpth.ckpt_dir) if fn.endswith('_model.pt')])
        latest = join(dpth.ckpt_dir, checkpoints[-1])
        print('\n------------------------Loading checkpoint {}'.format(basename(latest)))
        ckpt_loaded = torch.load(latest)
        model.load_state_dict(ckpt_loaded['model_state'])

        checkpoints = sorted([fn for fn in os.listdir(dpth.ckpt_dir) if fn.endswith('_geom_featmap.pt')])
        checkpoint = join(dpth.ckpt_dir, checkpoints[-1])
        load_latent_feats(checkpoint, geom_featmap)

        if args.mode.lower() == 'resume':
            optimizer.load_state_dict(ckpt_loaded['optimizer_state'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(DEVICE)
            epoch_now = ckpt_loaded['epoch'] + 1
            print('\n------------------------Resume training from epoch {}'.format(epoch_now))

        if 'test' in args.mode.lower():
            epoch_idx = ckpt_loaded['epoch']
            model.to(DEVICE)
            print('\n------------------------Test model with checkpoint at epoch {}'.format(epoch_idx))


    '''
    ------------ Training from scratch, or resume from saved checkpoints ------------
    '''
    if args.mode.lower() in ['train', 'resume']:

        train_set = CloDataSet(split='train', outfits=clo_body_assets.outfits['seen'], sample_spacing=args.data_spacing,
                               dataset_subset_portion=args.dataset_subset_portion, **dataset_args)

        val_outfit_name, val_outfit_idx = list(clo_body_assets.outfits['seen'].items())[0]
        val_outfit = {val_outfit_name: val_outfit_idx}
        val_set = CloDataSet(split='test', outfits=val_outfit, sample_spacing=args.data_spacing, dataset_subset_portion=1.0, **dataset_args)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

        writer = SummaryWriter(log_dir=dpth.log_dir)

        print("Total: {} training examples, {} val examples. Training started..".format(len(train_set), len(val_set)))

        model.to(DEVICE)
        start = time.time()
        pbar = range(epoch_now, n_epochs)
        for epoch_idx in pbar:
            wdecay_rgl = adjust_loss_weights(args.w_rgl, epoch_idx, mode='decay', start=args.decay_start, every=args.decay_every)
            wrise_normal = adjust_loss_weights(args.w_normal, epoch_idx,  mode='rise', start=args.rise_start, every=args.rise_every)
            w_reproj = adjust_loss_weights(args.w_reproj, epoch_idx, mode='decay', start=args.decay_start, every=args.decay_every, decay_rate=args.decay_rate)
            w_lbsw = adjust_loss_weights(args.w_lbsw, epoch_idx, mode='decay', start=args.decay_start, every=args.decay_every, decay_rate=args.decay_rate)

            if ((args.stop_lbsw_loss_at > 0) and (epoch_idx > args.stop_lbsw_loss_at)):
                w_reproj, w_lbsw = 0., 0.

            loss_weights = torch.tensor([args.w_s2m, args.w_m2s, wrise_normal, w_lbsw, w_reproj, wdecay_rgl, args.w_latent_rgl, args.w_corr_rgl])

            if ((epoch_idx > args.start_adaptive_at) and bool(args.adaptive_sample_in_training)):
                msg = 'Epoch {}, using adaptive sampling!'.format(epoch_idx)
                training_args['adaptive_sample_in_training'] = True
            
            if ((epoch_idx > args.start_adaptive_at) and bool(args.adaptive_weight_in_training)):
                msg = 'Epoch {}, using adaptive rgl wegiths!'.format(epoch_idx)
                training_args['adaptive_weight_in_training'] = True

            if ((epoch_idx > args.start_adaptive_at) and bool(args.adaptive_lbsw_weight_in_training)):
                msg = 'Epoch {}, using adaptive lbsw wegiths!'.format(epoch_idx)
                training_args['adaptive_lbsw_weight_in_training'] = True

            # do training for one epoch
            print('Epoch {}'.format(epoch_idx))
            train_stats = train(model, geom_featmap, train_loader, optimizer,
                                loss_weights=loss_weights,
                                **training_args)
            
            if epoch_idx % 50 == 0 or epoch_idx == n_epochs - 1:
                ckpt_path = join(dpth.ckpt_dir, '{}_epoch{}_model.pt'.format(exp_name, str(epoch_idx).zfill(5)))
                save_model(ckpt_path, model, epoch_idx, optimizer=optimizer)
                ckpt_path = join(dpth.ckpt_dir, '{}_epoch{}_geom_featmap.pt'.format(exp_name, str(epoch_idx).zfill(5)))
                save_latent_feats(ckpt_path, geom_featmap, epoch_idx)

            
            # test on val set every N epochs
            if epoch_idx % args.val_every == 0:
                dur = (time.time() - start) / (60 * (epoch_idx-epoch_now+1))
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                print('\n{}, Epoch {}, average {:.2f} min / epoch.'.format(dt_string, epoch_idx, dur))
                print('Weights s2m: {:.1e}, m2s: {:.1e}, normal: {:.1e}, lbsw: {:.1e}, rgl: {:.1e}'.format(args.w_s2m, args.w_m2s, wrise_normal, args.w_lbsw, wdecay_rgl))

                samples_dir_val = join(dpth.samples_dir_val_base, '{}_stage_{}'.format(args.stage, args.query_posmap_size), list(val_outfit.keys())[0])
                os.makedirs(samples_dir_val, exist_ok=True)
                val_stats = test_seen_clo(  
                                            model, 
                                            geom_featmap,
                                            val_loader, 
                                            epoch_idx,
                                            samples_dir_val,
                                            model_name=exp_name,
                                            save_all_results=bool(args.save_all_results),
                                            mode='val',
                                            **training_args
                                        )

                val_total_loss = np.stack(val_stats).dot(loss_weights)
                val_stats.append(np.array(val_total_loss))

                tensorboard_tabs = ['model2scan', 'scan2model', 'normal_loss', 'lbsw_loss', 'residual_square', 'latent_rgl', 'total_loss']
                stats = {'train': train_stats, 'val': val_stats}

                for split in ['train', 'val']:
                    for (tab, stat) in zip(tensorboard_tabs, stats[split]):
                        writer.add_scalar('{}/{}'.format(tab, split), stat, epoch_idx)

        end = time.time()
        t_total = (end - start) / 60
        print("Training finished, duration: {:.2f} minutes. Now eval on test set..\n".format(t_total))
        writer.close()


    '''
    ------------ Test model, seen outfits (SkiRT is outfit-specific model, so only test on seen outfits) ------------
    '''
    if args.mode.lower() in ['train', 'test', 'test_seen']:
        test_rst_msg = []
        test_rst_msg.append('\n\n{}, epoch={}, test query resolution={}, eval on body verts: {} \n'.format(exp_name, epoch_idx, args.query_posmap_size,  bool(args.eval_body_verts)))

        print('\n------------------------Eval on test data, seen outfits, unseen poses...')

        per_outfit_dataset = [{k:v} for k, v in clo_body_assets.outfits['seen'].items()]

        sum_chamfer_all_outfits, sum_normal_all_outfts, num_ex_all_outfits = 0, 0, 0

        test_rst_msg.append('\tEval on test set, seen clo:\n')

        training_args['query_posmap_size'] = 256
        for outfit in per_outfit_dataset: # outfit is a dict that contains a single key:val pair (a clothing type)

            test_set = CloDataSet(split='test', outfits=outfit, sample_spacing=args.data_spacing, dataset_subset_portion=1.0, **dataset_args)
            test_loader = DataLoader(test_set, batch_size=args.batch_size*2, shuffle=False, num_workers=4)
            
            samples_dir_outfit = join(dpth.samples_dir_test_seen_base, '{}_stage_{}'.format(args.stage, training_args['query_posmap_size']), list(outfit.keys())[0])
            os.makedirs(samples_dir_outfit, exist_ok=True)
            
            start = time.time()

            test_stats = test_seen_clo( 
                                        model, geom_featmap, test_loader, epoch_idx,
                                        samples_dir_outfit,
                                        mode='test_seen',
                                        model_name=exp_name,
                                        save_all_results=bool(args.save_all_results),
                                        **training_args
                                    )
            test_s2m, test_m2s, test_lnormal, test_lbsw_loss, test_reproj_loss, test_rgl, _, _ = test_stats


            # accumulate errors across all outfits
            sum_chamfer_outfit = (test_m2s+test_s2m) * len(test_set) 
            sum_normal_outfit = test_lnormal * len(test_set)

            sum_chamfer_all_outfits += sum_chamfer_outfit
            sum_normal_all_outfts += sum_normal_outfit
            num_ex_all_outfits += len(test_set)

            outfit_info = '{:<18}, {} examples.'.format(list(outfit.keys())[0], len(test_set))
            test_seen_result = "{:<34} m2s dist: {:.3e}, s2m dist: {:.3e}. Chamfer total: {:.3e}, normal loss: {:.3e}, lbsw loss: {:.3e}, reproj loss: {:.3e}, rgl term: {:.3e}.\n"\
                            .format(outfit_info, test_m2s, test_s2m, test_m2s+test_s2m, test_lnormal, test_lbsw_loss, test_reproj_loss, test_rgl)
            print(test_seen_result)
            test_rst_msg.append('\t\t{}'.format(test_seen_result))

            print('{} stage evaluation results saved to {}'.format(args.stage, samples_dir_outfit))
        
        # calculate the average error across all outfits
        avg_chamfer_all = sum_chamfer_all_outfits / num_ex_all_outfits
        avg_normal_all = sum_normal_all_outfts / num_ex_all_outfits
        test_seen_full_stats = '\t\tOn all seen data, {} exmaples, average Chamfer: {:.3e}, average normal loss: {:.3e}\n'\
            .format(num_ex_all_outfits, avg_chamfer_all, avg_normal_all)
        test_rst_msg.append(test_seen_full_stats)

        with open(join(PROJECT_DIR, 'results', 'eval_results.txt'), 'a+') as fp:
            fp.writelines(test_rst_msg)




if __name__ == '__main__':
    main()