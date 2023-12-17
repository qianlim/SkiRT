import os
from os.path import join, dirname, realpath

import torch
import torch.nn.functional as F
import numpy as np

SCRIPT_DIR = dirname(realpath(__file__))


def getIdxMap_torch(img, offset=False):
    # img has shape [channels, H, W]
    C, H, W = img.shape
    import torch
    idx = torch.stack(torch.where(~torch.isnan(img[0])))
    if offset:
        idx = idx.float() + 0.5
    idx = idx.view(2, H * W).float().contiguous()
    idx = idx.transpose(0, 1)

    idx = idx / (H-1) if not offset else idx / H
    return idx


def load_masks(PROJECT_DIR, posmap_size, body_model='smpl'):
    uv_mask_faceid = np.load(join(PROJECT_DIR, 'assets', 'uv_masks', 'uv_mask{}_with_faceid_{}.npy'.format(posmap_size, body_model))).reshape(posmap_size, posmap_size)
    uv_mask_faceid = torch.from_numpy(uv_mask_faceid).long().cuda()
    
    smpl_faces = np.load(join(PROJECT_DIR, 'assets', '{}_faces.npy'.format(body_model.lower()))) # faces = triangle list of the body mesh
    flist = torch.tensor(smpl_faces.astype(np.int32)).long()
    flist_uv = get_face_per_pixel(uv_mask_faceid, flist).cuda() # Each (valid) pixel on the uv map corresponds to a point on the SMPL body; flist_uv is a list of these triangles

    points_idx_from_posmap = (uv_mask_faceid!=-1).reshape(-1)

    uv_coord_map = getIdxMap_torch(torch.rand(3, posmap_size, posmap_size)).cuda()
    uv_coord_map.requires_grad = True

    return flist_uv, points_idx_from_posmap, uv_coord_map


def load_barycentric_coords(PROJECT_DIR, posmap_size, body_model='smpl'):
    '''
    load the barycentric coordinates (pre-computed and saved) of each pixel on the positional map.
    Each pixel on the positional map corresponds to a point on the SMPL / SMPL-X body (mesh)
    which falls into a triangle in the mesh. This function loads the barycentric coordinate of the point in that triangle.
    '''
    bary = np.load(join(PROJECT_DIR, 'assets', 'bary_coords_uv_map', 'bary_coords_{}_uv{}.npy'.format(body_model, posmap_size)))
    bary = bary.reshape(posmap_size, posmap_size, 3)
    return torch.from_numpy(bary).cuda()


def get_face_per_pixel(mask, flist):
    '''
    :param mask: the uv_mask returned from posmap renderer, where -1 stands for background
                 pixels in the uv map, where other value (int) is the face index that this
                 pixel point corresponds to.
    :param flist: the face list of the body model,
        - smpl, it should be an [13776, 3] array
        - smplx, it should be an [20908,3] array
    :return:
        flist_uv: an [img_size, img_size, 3] array, each pixel is the index of the 3 verts that belong to the triangle
    Note: we set all background (-1) pixels to be 0 to make it easy to parralelize, but later we
        will just mask out these pixels, so it's fine that they are wrong.
    '''
    mask2 = mask.clone()
    mask2[mask == -1] = 0 #remove the -1 in the mask, so that all mask elements can be seen as meaningful faceid
    flist_uv = flist[mask2]
    return flist_uv


def get_scan_pcl_by_name(target_basename, num_pts=40000, dataset_type='resynth'):
    '''
    given the 'target name' (i.e. the name of a packed frame) from the dataloader, find its gt mesh/dense point cloud, 
    and sample a point cloud at the specified resolution from it as the target of the optimization in the single scan animation application.

    args:
        target_basename: the name of a frame in the packed data. In general contains info of <subject> <sequence_name> <frame_id>.
            Examples:
                # CAPE data: 03375_shortlong_ATUsquat_trial2.000009
                # ReSynth data: rp_rp_aaron_posed_002.96_jerseyshort_hips.00020
        num_pts: the number of points to sample from the scan surface (if the scan is a mesh as in CAPE) or to subsample from
                 the dense GT point cloud scan (if the scan is a point cloud as in ReSynth)
        dataset_type: 'cape' or 'resynth', will decide which body model to use.

    returns:
        scan_pc: a sampled point cloud, treated as the 'ground truth scan' for the optimization
        scan_normal: the points' normals of the sampled point cloud
    '''
    import numpy as np
    import trimesh
    import open3d as o3d
    from os.path import join

    scan_data_root = join(SCRIPT_DIR, '../data', dataset_type.lower(), 'scans')

    if dataset_type.lower() == 'cape':
        faces = np.load(join(SCRIPT_DIR, '../assets', 'smpl_faces.npy'))

        seq_name, frame_id = target_basename.split('.')
        subj_id, clo_seq = seq_name.split('_', 1)
        if subj_id == '03375':
            subj_id = '{}_foot_corrected'.format(subj_id)
        scan_data_fn = join(scan_data_root, subj_id, clo_seq, '{}.{}.npz'.format(clo_seq, frame_id))
        
        # the 'scan' data is the smpl registration of cape raw scans. load it and sample points on its surface
        # we sample point clouds here and treat it as scan (discard the connectivity info)
        scan_verts = np.load(scan_data_fn)['v_posed'] 
        scan = trimesh.Trimesh(vertices=scan_verts, faces=faces, process=False)

        scan_pc, faceid = trimesh.sample.sample_surface_even(scan, num_pts) # sample_even may cause smaller number of points sampled than wanted
        scan_normals = scan.face_normals[faceid]
        scan_pc = torch.tensor(scan_pc).float().unsqueeze(0).cuda()
        scan_normals = torch.tensor(scan_normals).float().unsqueeze(0).cuda()

        return scan_pc, scan_normals

    else:
        outfit, pose, frame_id = target_basename.split('.')
        outfit_basename, suboutfit_id = outfit[3:].rsplit('_', 1)
        suboutfit_id = suboutfit_id.replace('suboutfit', '')

        scan_data_fn = join(scan_data_root, outfit[3:], pose, '{}_pcl.ply'.format(frame_id)) # [3:]: for resynth data the name is rp_rp_<subj_name>_xxxx. first 3 are redundant characters

        scan = o3d.io.read_point_cloud(scan_data_fn)
        points = np.array(scan.points)
        normals = np.array(scan.normals)

        randperm = np.random.permutation(len(points))
        points = points[randperm[:num_pts]]
        normals = normals[randperm[:num_pts]]

        scan_pc = torch.tensor(points).float().unsqueeze(0).cuda()
        scan_normals = torch.tensor(normals).float().unsqueeze(0).cuda()
        return scan_pc, scan_normals


def save_latent_feats(filepath, latent_vec, epoch):
    
    if not isinstance(latent_vec, torch.Tensor):
        all_latents = latent_vec.state_dict()
    else:
        all_latents = latent_vec

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(filepath),
    )
    

def load_latent_feats(filepath, lat_vecs):
    full_filename = filepath

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)
    if isinstance(data["latent_codes"], torch.Tensor):
        lat_vecs.data[...] = data["latent_codes"].data[...]
    else:
        raise NotImplementedError

    return data["epoch"]


def save_model(path, model, epoch, optimizer=None):
    model_dict = {
            'epoch': epoch,
            'model_state': model.state_dict()
        }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()
    print('Saving checkpoint to {}'.format(path))
    torch.save(model_dict, path)


def tensor2numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()


def customized_export_ply(outfile_name, v, f = None, v_n = None, v_c = None, f_c = None, e = None):
    '''
    Author: Jinlong Yang, jyang@tue.mpg.de

    Exports a point cloud / mesh to a .ply file
    supports vertex normal and color export
    such that the saved file will be correctly displayed in MeshLab

    # v: Vertex position, N_v x 3 float numpy array
    # f: Face, N_f x 3 int numpy array
    # v_n: Vertex normal, N_v x 3 float numpy array
    # v_c: Vertex color, N_v x (3 or 4) uchar numpy array
    # f_n: Face normal, N_f x 3 float numpy array
    # f_c: Face color, N_f x (3 or 4) uchar numpy array
    # e: Edge, N_e x 2 int numpy array
    # mode: ascii or binary ply file. Value is {'ascii', 'binary'}
    '''

    v_n_flag=False
    v_c_flag=False
    f_c_flag=False
    
    N_v = v.shape[0]
    assert(v.shape[1] == 3)
    if not type(v_n) == type(None):
        assert(v_n.shape[0] == N_v)
        if type(v_n) == 'torch.Tensor':
            v_n = v_n.detach().cpu().numpy()
        v_n_flag = True
    if not type(v_c) == type(None):
        assert(v_c.shape[0] == N_v)
        v_c_flag = True
        if v_c.shape[1] == 3:
            # warnings.warn("Vertex color does not provide alpha channel, use default alpha = 255")
            alpha_channel = np.zeros((N_v, 1), dtype = np.ubyte)+255
            v_c = np.hstack((v_c, alpha_channel))

    N_f = 0
    if not type(f) == type(None):
        N_f = f.shape[0]
        assert(f.shape[1] == 3)
        if not type(f_c) == type(None):
            assert(f_c.shape[0] == f.shape[0])
            f_c_flag = True
            if f_c.shape[1] == 3:
                # warnings.warn("Face color does not provide alpha channel, use default alpha = 255")
                alpha_channel = np.zeros((N_f, 1), dtype = np.ubyte)+255
                f_c = np.hstack((f_c, alpha_channel))
    N_e = 0
    if not type(e) == type(None):
        N_e = e.shape[0]

    with open(outfile_name, 'w') as file:
        # Header
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex %d\n'%(N_v))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')

        if v_n_flag:
            file.write('property float nx\n')
            file.write('property float ny\n')
            file.write('property float nz\n')
        if v_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        file.write('element face %d\n'%(N_f))
        file.write('property list uchar int vertex_indices\n')
        if f_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        if not N_e == 0:
            file.write('element edge %d\n'%(N_e))
            file.write('property int vertex1\n')
            file.write('property int vertex2\n')

        file.write('end_header\n')

        # Main body:
        # Vertex
        if v_n_flag and v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2], \
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        elif v_n_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2]))
        elif v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        else:
            for i in range(0, N_v):
                file.write('%f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2]))
        # Face
        if f_c_flag:
            for i in range(0, N_f):
                file.write('3 %d %d %d %d %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2],\
                    f_c[i,0], f_c[i,1], f_c[i,2], f_c[i,3]))
        else:
            for i in range(0, N_f):
                file.write('3 %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2]))

        # Edge
        if not N_e == 0:
            for i in range(0, N_e):
                file.write('%d %d\n'%(e[i,0], e[i,1]))


def vertex_normal_2_vertex_color(vertex_normal):
    # Normalize vertex normal
    import torch
    if torch.is_tensor(vertex_normal):
        vertex_normal = vertex_normal.detach().cpu().numpy()
    normal_length = ((vertex_normal**2).sum(1))**0.5
    normal_length = normal_length.reshape(-1, 1)
    vertex_normal /= normal_length
    # Convert normal to color:
    color = vertex_normal * 255/2.0 + 128
    return color.astype(np.ubyte)


def draw_correspondence(pcl_1, pcl_2, output_file):
    '''
    Given a pair of (minimal, clothed) point clouds which have same #points,
    draw correspondences between each point pair as a line and export to a .ply
    file for visualization.
    '''
    assert(pcl_1.shape[0] == pcl_2.shape[0])
    N = pcl_2.shape[0]
    v = np.vstack((pcl_1, pcl_2))
    arange = np.arange(0, N)
    arange = arange.reshape(-1,1)
    e = np.hstack((arange, arange+N))
    e = e.astype(np.int32)
    customized_export_ply(output_file, v, e = e)

def save_result_examples(save_dir, model_name, result_name, points,
                         normals=None, patch_color=None, 
                         texture=None, coarse_pts=None, lbsw=None,
                         gt=None, epoch=None, is_cano=False, vert_connectivity=None,
                         extra_suffix=''):
    # works on single pcl, i.e. [#num_pts, 3], no batch dimension
    from os.path import join
    import numpy as np

    cano_suffix = '_cano' if is_cano else ''

    if epoch is None:
        normal_fn = '{}_{}_pred{}{}.ply'.format(model_name,result_name, cano_suffix, extra_suffix)
    else:
        normal_fn = '{}_epoch{}_{}_pred{}{}.ply'.format(model_name, str(epoch).zfill(4), result_name, cano_suffix, extra_suffix)
    normal_fn = join(save_dir, normal_fn)
    points = tensor2numpy(points)
    
    if normals is not None:
        normals = tensor2numpy(normals)
        color_normal = vertex_normal_2_vertex_color(normals)
        customized_export_ply(normal_fn, v=points, f=vert_connectivity, v_n=normals, v_c=color_normal)

    if lbsw is not None:
        from lib.utils_vis import color_lbsw
        
        # assumes lbsw in [num_pts, num_joints], i.e. channels last
        color = color_lbsw(lbsw, mode='diffuse', shuffle_color=True)
        lbsw_fn = normal_fn.replace('pred{}{}.ply'.format(cano_suffix, extra_suffix), 'pred_lbsw{}{}.ply'.format(cano_suffix, extra_suffix))
        customized_export_ply(lbsw_fn, v=points, f=vert_connectivity, v_c=color)

    if patch_color is not None:
        patch_color = tensor2numpy(patch_color)
        if patch_color.max() < 1.1:
            patch_color = (patch_color*255.).astype(np.ubyte)
        pcolor_fn = normal_fn.replace('pred.ply', 'pred_patchcolor.ply')
        customized_export_ply(pcolor_fn, v=points, v_c=patch_color)
    
    if texture is not None:
        texture = tensor2numpy(texture)
        if texture.max() < 1.1:
            texture = (texture*255.).astype(np.ubyte)
        texture_fn = normal_fn.replace('pred.ply', 'pred_texture.ply')
        customized_export_ply(texture_fn, v=points, v_c=texture)

    if coarse_pts is not None:
        coarse_pts = tensor2numpy(coarse_pts)
        coarse_fn = normal_fn.replace('pred.ply', 'interm.ply')
        customized_export_ply(coarse_fn, v=coarse_pts)

    if gt is not None: 
        gt = tensor2numpy(gt)
        gt_fn = normal_fn.replace('pred.ply', 'gt.ply')
        customized_export_ply(gt_fn, v=gt)



def save_obj_mesh(mesh_path, verts, faces=None, color=None):
    file = open(mesh_path, 'w')
    for i, v in enumerate(verts):
        if color is None:
            file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        else:
            file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], color[i][0], color[i][1], color[i][2]))
    if faces is not None:
        for f in faces:
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm

# compute tangent and bitangent
def compute_tangent(vertices, faces, normals, uvs, faceuvs):    
    # NOTE: this could be numerically unstable around [0,0,1]
    # but other current solutions are pretty freaky somehow
    c1 = np.cross(normals, np.array([0,1,0.0]))
    tan = c1
    normalize_v3(tan)
    btan = np.cross(normals, tan)
    return tan, btan


def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces


def load_config(path):
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

