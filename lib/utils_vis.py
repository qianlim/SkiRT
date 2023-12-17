from lib.utils_io import customized_export_ply

def color_lbsw(lbsw=None, mode='discrete', color_mode=255, shuffle_color=False):
    '''
    lbsw: [Nv, J]
    visualize skinning weight distribution on mesh surface by coloring them
    mode: 'most_influence": paint the surface with the joint that influences the region the most
          'diffuse': paint the surface using colors diffused with the lbsw definition
    '''

    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    num_verts, num_joints = lbsw.shape
    if num_joints == 22:
        perm =[ 3, 10, 15, 20, 13, 18,  8, 12, 16,  6,  7, 14, 17,  4,  2,  0,  9, 19, 11,  1, 21,  5]
    elif num_joints == 24:
        perm = [ 3, 10, 15, 20, 13, 18, 23, 8, 12, 16,  6, 22, 7, 14, 17,  4,  2,  0,  9, 19, 11,  1, 21, 5]
    else:
        raise NotImplementedError('Only 22 joints (smplx) or 24 joints (smpl) are supported for shuffling colors')

    cmap = plt.cm.get_cmap("hsv", num_joints)
    lbsw = lbsw.float().cpu()

    if mode != 'diffuse':
        biggest_joint_idx = torch.argmax(lbsw, dim=1).numpy()
        color = [cmap(x) for x in biggest_joint_idx]
        color = np.array(color)[:,:3] # rgba->rgb
        
    else:
        cmap_val = [cmap(x) for x in range(num_joints)]
        cmap_val = torch.tensor(cmap_val)[:, :3].float() # rgba->rgb, [num joints, 3] 
        if shuffle_color:
            perm = np.array(perm)
            cmap_val = cmap_val[perm]
        color = torch.einsum('ij,jk->ik', lbsw, cmap_val).numpy()
    
    if color_mode == 255:
        color = (color*255).astype(np.uint8)
    return color # [num verts, 3]


def vis_skinning_weights(mode='smpl', vis_screen=True):
    if mode.lower() in ['smpl', 'smplx']:
        import smplx
        from psbody.mesh import Mesh

        from lib_data.data_paths import DataPaths
        dpth = DataPaths
        
        model = smplx.create(model_path=dpth.smpl_path, model_type=mode)
        model_output = model()

        lbsw = model.lbs_weights[:,:24]

        color = color_lbsw(lbsw, mode='diffuse')

        if vis_screen:
            mesh = Mesh(model_output.vertices.squeeze().detach().numpy(), model.faces)
            mesh.set_vertex_colors(color)
            mesh.show()

        return color


def save_vis_pred_lbsw(query_locs, lbsw):
    '''
    channel first
    [B, 3, N]
    [B, 24, N]
    '''
    from lib.utils_io import customized_export_ply
    colors = [color_lbsw(x, mode='diffuse') for x in lbsw.permute(0, 2, 1).detach().cpu()]
    vv = query_locs.permute(0, 2, 1).detach().cpu().numpy()
    for i in range(len(vv)):
        customized_export_ply("./vis/lbs_{}.ply".format(i), vv[i], v_c=colors[i])



def vis_pcl(pcl, num_vis=5):
    '''
    pcl: [B, N_pts, 3]
    '''
    import numpy as np
    import torch
    from psbody.mesh import Mesh
    
    if isinstance(pcl, torch.Tensor):
        pcl = pcl.detach().cpu().numpy()
    
    if len(pcl) < num_vis:
        pcl_to_vis = pcl
    else:
        spacing = len(pcl) // num_vis
        pcl_to_vis = pcl[::spacing]
    
    for x in pcl_to_vis:
        Mesh(x).show()