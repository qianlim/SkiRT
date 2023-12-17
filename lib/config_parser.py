from os.path import join, basename, dirname, realpath
import sys

PROJECT_DIR = join(dirname(realpath(__file__)), '..')
sys.path.append(PROJECT_DIR)


def parse_config(argv=None):
    import configargparse
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.DefaultConfigFileParser
    description = 'articulated bps project'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='POP')

    # general settings                              
    parser.add_argument('--config', is_config_file=True, default='{}/configs/model_dev.yaml'.format(PROJECT_DIR), help='config file path')
    parser.add_argument('--model_config', default='{}/configs/model_config.yaml'.format(PROJECT_DIR), help='model architecture config file path')
    parser.add_argument('--name', type=str, default='debug', help='name of a model/experiment. this name will be used for saving checkpoints and will also appear in saved examples')
    parser.add_argument('--name_coarse', type=str, default='', help='for two stage training, specify the coarse stage trained model name to load the corresponding coarse preds')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'resume', 'test', 'test_seen', 'test_unseen'], help='train/resume training/test, \
                                  for test_seen will evaluate on seen outfits unseen poses; test_unseen will test on unseen outfits (specified in configs/clo_config.yaml')
    parser.add_argument('--outfit_name', type=str, default='rp_anna_posed_001', help='the subject+outfit name. \
                                  for CAPE it is like 03375_blazerlong; for resynth it is like rp_anna_posed_001')
    parser.add_argument('--stage', type=str, default='coarse', choices=['coarse', 'fine'], help='coarse or fine stage. Fine stage requires an existing trained coarse stage')

    # architecture related
    parser.add_argument('--hsize', type=int, default=256, help='hideen layer size of the ShapeDecoder mlp')
    parser.add_argument('--nf', type=int, default=32)
    parser.add_argument('--use_dropout', type=int, default=0, help='whether use dropout in the UNet')
    parser.add_argument('--up_mode', type=str, default='upconv',  choices=['upconv', 'upsample'], help='the method to upsample in the UNet')
    parser.add_argument('--latent_size', type=int, default=256, help='the size of a latent vector that conditions the unet, leave it untouched (it is there for historical reason)')
    parser.add_argument('--pix_feat_dim', type=int, default=64, help='dim of the pixel-wise latent code output by the UNet')
    parser.add_argument('--pos_encoding', type=int, default=0, help='use Positional Encoding (PE) for uv coords instead of plain concat')
    parser.add_argument('--posemb_incl_input', type=int, default=1, help='if use PE, then include original coords in the positional encoding')
    parser.add_argument('--num_emb_freqs', type=int, default=6, help='if use PE: number of frequencies used in the positional embedding')
    parser.add_argument('--c_geom', type=int, default=64, help='channels of the geometry feature map')
    parser.add_argument('--c_pose', type=int, default=64, help='dim of the pixel-wise latent code output by the pose Unet')
    parser.add_argument('--transf_scaling', type=float, default=0.02, help='scale the transformation matrix (empirically found: will slightly improve performance')

    # body-modeling techniques related
    parser.add_argument('--pose_input', type=str, default='posmap', choices=['posmap', 'verts'], 
                        help='the format of the posed body to use as pose input (fine stage only). options: 2d positional map (posmap), or vertices of unclothed bodies (verts)')
    parser.add_argument('--pose_feat_type', type=str, default='conv', choices=['scanimate', 'conv', 'both'], help='pose feature type as input to the \
                        shape decoder. If scanimate, use the scanimate-style filtered pose parameters as pose feature; if conv, use the unet output from the posed body uv map as pose feature.')
    parser.add_argument('--posemap_type', type=str, default='both', help='neighborhood definition in the kinematic tree. options: parent, child or both. Used in building scanimate-like pose features.')
    parser.add_argument('--normalize_posemap', type=int, default=1, help='1 for normalize the pose filtering matrix, 0 for not')
    parser.add_argument('--n_traverse', type=int, default=4, help='range of neighborhoods to filter pose features. Used in building scanimate-like pose features.')

    # data related
    parser.add_argument('--dataset_type', type=str, default='cape', help='cape or resynth. for CAPE, will use SMPL in the code, for ReSynth will use SMPL-X.')
    parser.add_argument('--data_spacing', type=int, default=1, help='get every N examples from dataset (set N a large number for fast experimenting)')
    parser.add_argument('--query_posmap_size', type=int, default=256, help='size of the **query** UV positional map')
    parser.add_argument('--inp_posmap_size', type=int, default=128, help='size of UV positional **feature** map')
    parser.add_argument('--scan_npoints', type=int, default=-1, help='number of points used in the GT point set. By default -1 will use all points (40000);\
                                                                      setting it to another number N will randomly sample N points at each iteration as GT for training.')
    parser.add_argument('--dataset_subset_portion', type=float, default=1.0, help='the portion with which a subset from all training data is randomly chosen, value between 0 and 1 (for faster training)')
    parser.add_argument('--random_subsample_scan', type=int, default=0, help='wheter use the full dense scan point cloud as the GT for the optimization in the test-unseen-outfit scenario,\
                                                                              or randomly sample a subset of points from it at every optimization iteration')

    # loss func related
    parser.add_argument('--w_m2s', type=float, default=1e4, help='weight for the Chamfer loss part 1: (m)odel to (s)can, i.e. from the prediction to the GT points')
    parser.add_argument('--w_s2m', type=float, default=1e4, help='weight for the Chamfer loss part 2: (s)can to (m)odel, i.e. from the GT points to the predicted points')
    parser.add_argument('--w_normal', type=float, default=1.0, help='weight for the normal loss term')
    parser.add_argument('--w_rgl', type=float, default=2e3, help='weight for residual length regularization term')
    parser.add_argument('--w_latent_rgl', type=float, default=1.0, help='weight for regularization term on the geometric feature tensor')
    parser.add_argument('--w_lbsw', type=float, default=5e1, help='weight for regularization term on the lbsw weights (deviation from smpl lbsw)')
    parser.add_argument('--w_reproj', type=float, default=5e2, help='weight for regularization term on the lbsw, diff of the reprojected minimal cano body from the minimal posed body')
    parser.add_argument('--w_corr_rgl', type=float, default=1.0, help='weight for the rgl term for the variance of predictions across a batch')

    # training / eval related
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--decay_start', type=int, default=100, help='start to decay the regularization loss term from the X-th epoch')
    parser.add_argument('--rise_start', type=int, default=100, help='start to rise the normal loss term from the X-th epoch')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--decay_every', type=int, default=400, help='decaly the regularization loss weight every X epochs')
    parser.add_argument('--rise_every', type=int, default=400, help='rise the normal loss weight every X epochs')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='decay by this factor at every X epochs')
    parser.add_argument('--stop_lbsw_loss_at', type=int, default=-1, help='stop the lbsw loss at the X-th epoch')
    parser.add_argument('--val_every', type=int, default=20, help='validate every x epochs')
    parser.add_argument('--eval_body_verts', type=int, default=0, help='evaluate using SMPL vertices as query points, instead of the regular uv pixel grid.')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--lr_geomfeat', type=float, default=5e-4, help='learning rate for the geometric feature tensor auto-decoding')
    parser.add_argument('--save_all_results', type=int, default=0, help='save the entire test set results at inference')
    parser.add_argument('--query_xyz', type=int, default=0, help='use uv (2d) or xyz (3d) as query coords of the points')
    parser.add_argument('--use_vert_geom_feat', type=int, default=0, help='use geom feat that are anchored to SMPL verts, instead of on UV maps')
    parser.add_argument('--use_global_geom_feat', type=int, default=0, help='use a long global vector for geom feats instead of local feats')
    parser.add_argument('--use_hires_smpl', type=int, default=0, help='use hires smpl/smplx for the geom feat if define geom feat on verts')
    parser.add_argument('--incl_query_nml', type=int, default=0, help='if query with xyz, whether include the smpl body normal of the query points')
    parser.add_argument('--use_pose_emb', type=int, default=0, help='use a small network after pose params')
    parser.add_argument('--use_jT', type=int, default=0, help='use jT or pre-computed vtransf. jT is per body joint 4x4 transf matrix exported from SMPL/SMPLX')
    parser.add_argument('--pred_lbsw', type=int, default=0, help='pred lbsw or use the gt smpl lbsw')
    parser.add_argument('--transf_only_disp', type=int, default=1)
    parser.add_argument('--use_pre_diffuse_lbsw', type=int, default=0, help='use the pre-diffused lbsw, otherwise use smpl lbsw')
    parser.add_argument('--use_variance_rgl', type=int, default=0, help='whether regularize the predictions of the same garment across the frames (to enforce correspondence). \
                                                                        if yes, will use the custom data loader for training.')
    parser.add_argument('--use_raw_scan', type=int, default=0, help='CAPE data only, using raw scans (noisy) as GT instead of the clean registered meshes as GT')                                                                
    parser.add_argument('--single_direc_chamfer', type=int, default=0, help='CAPE raw scan data only, use single directional chamfer loss for the raw scan training (as raw scans has holes and missing regions)')    
    parser.add_argument('--exclude_handfeet', type=int, default=0, help='CAPE raw scan data only. Exclude supervision on hand and feet points as they are missing in most raw scans.')    

    # training / eval related, part 2: adaptive sampling of points during train/eval
    parser.add_argument('--num_pt_adaptve', type=int, default=5000)
    parser.add_argument('--adaptive_sample_loops', type=int, default=0)
    parser.add_argument('--num_pt_random_train', type=int, default=0)
    parser.add_argument('--adaptive_sample_in_training', type=int, default=0, help='use adaptive sampling during training')
    parser.add_argument('--adaptive_weight_in_training', type=int, default=0, help='use adaptive per point regularization weights during training')
    parser.add_argument('--adaptive_lbsw_weight_in_training', type=int, default=0, help='use adaptive per point lbsw weights during training')
    parser.add_argument('--start_adaptive_at', type=int, default=3, help='use adaptive per point lbsw weights during training')
    parser.add_argument('--start_adaptive_rgl_at', type=int, default=150, help='use adaptive per point lbsw weights during training')
    parser.add_argument('--start_adaptive_lbsw_at', type=int, default=10, help='use adaptive per point lbsw weights during training')
    parser.add_argument('--use_original_grid', type=int, default=0, help='the uv grid as in original POP')

    args, _ = parser.parse_known_args()

    return args


def load_net_config(path):
    ''' Loads config file.
    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    import yaml
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg