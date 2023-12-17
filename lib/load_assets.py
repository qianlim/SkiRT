from os.path import join, exists
import glob
import numpy as np
import torch
import smplx
from psbody.mesh import Mesh

from lib.utils_io import load_masks, load_barycentric_coords, load_obj_mesh
from lib.utils_model import FeatureVolume
from lib.utils_model import get_body_lbsw, get_body_vert_bary
from lib.utils_pose import get_posemap


class BodyModelAssets():
    def __init__(self, model_args, data_paths, device):
        project_dir = data_paths.project_dir

        body_model_type = 'smpl' if model_args.dataset_type.lower() == 'cape' else 'smplx'
        self.body_model_type = body_model_type
        self.bm = smplx.create(model_path=data_paths.smpl_path, model_type=body_model_type)

        is_hires = '_hires' if model_args.use_hires_smpl else ''
        max_joint_id_clo_related = 23 if self.body_model_type == 'smpl' else 21 # doesn't include root joint here; smplx doesn't have hand joint, just use the wrist

        self.cano_v, self.cano_f, self.cano_vt, self.cano_ft = load_obj_mesh(join(project_dir, 'assets/template_mesh_{}{}.obj'.format(self.body_model_type, is_hires)), with_texture=True)
        self.cano_vn = Mesh(self.cano_v, self.cano_f).estimate_vertex_normals()

        # positinal map of smpl template body in canonical pose
        cano_posmap = np.load(join(project_dir, 'assets', '{}_posmap_128.npy'.format(self.body_model_type))).reshape(128,128,3)
        self.cano_posmap = torch.tensor(cano_posmap)[None,:].permute(0,3,1,2).to(device)
        
        self.vert_lbsw = get_body_lbsw(body_model_loaded=self.bm, hires=bool(model_args.use_hires_smpl)) # lbs weights on the underlying body verts
        self.pose_map = get_posemap(model_args.posemap_type, max_joint_id_clo_related+1, self.bm.parents, model_args.n_traverse, bool(model_args.normalize_posemap)) # the pose feat filter matrix W in SCANimate Eq 18
        
        # barycentric coords and the face id of the vertices on the body mesh, used for coarse stage training
        self.vert_bary, self.vert_fid = get_body_vert_bary(Mesh(self.cano_v, self.cano_f), device=device)

        # uv locations, indices of the valid pixels and uv coordinates on the **query** (high-res) UV map
        self.flist_uv, self.valid_idx, self.uv_coord_map = load_masks(project_dir, model_args.query_posmap_size, body_model=body_model_type)
        self.bary_coords = load_barycentric_coords(project_dir, model_args.query_posmap_size, body_model=body_model_type)

        query_body_cano = Mesh(filename=join(project_dir, 'assets', 'query_locs_3d_{}_pop_grid_{}.ply'.format(body_model_type, model_args.query_posmap_size)))
        self.query_locs_3d = torch.tensor(query_body_cano.v).float().to(device)
        self.query_nml = torch.tensor(query_body_cano.vn).float().to(device) # normal vec at the query locations (on the minimal body)

        # when dealing with cape dataset, we exclude the hand and feet regions from the loss computation
        if (bool(model_args.exclude_handfeet) and (model_args.dataset_type.lower() == 'cape')):
            non_handfeet_mask = np.load(join(project_dir, 'assets', 'non_handfeet_pt_mask_{}.npy'.format(body_model_type)))
            self.non_handfeet_mask = torch.tensor(non_handfeet_mask).float().to(device)
        else:
            self.non_handfeet_mask = None

        # ! SkiRT is subject specific, but here we keep the outfit names disct consistent with PoP
        #  for future extension to multi-subject models. Therefore 'seen' category has only 1 and
        #  there's no 'unseen' outfits here.
        self.outfits = {
            'seen': {model_args.outfit_name: 0},
            'unseen': {}
        }
        self.num_outfits_seen, self.num_outfits_unseen = len(self.outfits['seen']), len(self.outfits['unseen'])

        if model_args.stage == 'fine':
            print('Loading pre-trained coarse shape...\n')
            # load pre-diffused volume LBSW
            vol_lbsw = torch.load(join(project_dir, 'assets','pre_diffused_lbsw', 'optimized_lbsw_kernel5_{}.pt'.format(body_model_type))).detach().to(device)
            vol_lbsw_sampler = FeatureVolume(vol_values=vol_lbsw)

            # get coarse stage shape prediction, and then use coarse shape points to query pre-diffused LBSW field
            # to get the init LBSW for the fine stage
            coarse_pred_basedir = join(data_paths.samples_path, model_args.name, 'test_seen', 'coarse_stage_256')
            outfit_names = list(self.outfits['seen'].keys())

            num_regular_pts = 47911 if body_model_type=='smplx' else 50701
            diffused_lbsws = torch.empty(len(outfit_names), num_regular_pts, max_joint_id_clo_related+1)
            coarse_shapes = torch.empty(len(outfit_names), num_regular_pts, 3)

            
            for outfit_name in outfit_names:
                outfit_id = self.outfits['seen'][outfit_name]
                if not exists(join(coarse_pred_basedir, outfit_name)):
                    raise ValueError('Coarse stage predictions not found at {}. Please get coarse shape preds first.'.format(join(coarse_pred_basedir, outfit_name)))
            
                coarse_pred_meanshape_fn = sorted(glob.glob(join(coarse_pred_basedir, outfit_name, '*pred_cano_mean.ply')))[0]
                
                coarse_pred_fn = sorted(glob.glob(join(coarse_pred_basedir, outfit_name, '*pred_cano.ply')))[0]
                
                coarse_pred = torch.tensor(Mesh(filename=coarse_pred_fn).v).float().cuda()
                coarse_pred_mean = torch.tensor(Mesh(filename=coarse_pred_meanshape_fn).v).float().cuda()
                
                # use the predicted coarse clothed body to query the pre-diffused LBS weight field as init for the fine stage
                lbsw_outfit = vol_lbsw_sampler(coarse_pred_mean)

                diffused_lbsws[outfit_id, :] = lbsw_outfit # [N_query_pts, num_bones]
                coarse_shapes[outfit_id, :] = coarse_pred
            self.coarse_shapes = coarse_shapes.to(device)
            self.diffused_lbsws = diffused_lbsws.to(device) if bool(model_args.use_pre_diffuse_lbsw) else None

        else:
            self.coarse_shapes = None
            self.diffused_lbsws = None

        if model_args.use_hires_smpl: # we provide an upsampled version of SMPL; could be useful if use per-vertex features (not used in paper)
            bary_data = np.load('assets/barycoords_{}_hires_verts.npz'.format(self.body_model_type))
            hires_assets = {}
            hires_assets['bary'] = torch.tensor(bary_data['bary_coords']).float().to(device)
            hires_assets['faceid'] = torch.tensor(bary_data['face_ids']).long().to(device)
            hires_assets['face_original'] = torch.tensor(self.bm.faces.astype(np.int32)).long().to(device)
        else:
            hires_assets = None
        self.hires_assets = hires_assets