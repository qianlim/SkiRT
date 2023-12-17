'''
methods for diffusing the lbs weights from the body mesh to R3 according to a certain distance function
'''
import os
from os.path import join, dirname, realpath
import sys
PROJECT_DIR = join(dirname(realpath(__file__)), '..')
sys.path.append(PROJECT_DIR)

import numpy as np
import torch
from psbody.mesh import Mesh
import smplx
from tqdm import tqdm


from lib.utils_io import load_obj_mesh, customized_export_ply
from lib.utils_model import get_body_lbsw
from lib.modules import GaussianSmoothing
from lib.utils_vis import color_lbsw



class DiffuseLBSW(object):
    def __init__(self, body_model_name='smplx', is_hires_smpl=False, grid_resl=16, blend_weight_pow=1):
        super().__init__()
        from lib_data.data_paths import DataPaths
        dpth = DataPaths()
        
        self.body_model = smplx.create(model_path=dpth.smpl_path, model_type=body_model_name)

        is_hires = '_hires' if is_hires_smpl else ''
        cano_v, f = load_obj_mesh('assets/template_mesh_{}{}.obj'.format(body_model_name, is_hires))

        self.cano_v = torch.tensor(cano_v).float()
        self.f = torch.tensor(f.astype(np.int32)).long()

        self.grid_resl = grid_resl
        self.thres = 0.05 # 0.1 m
        self.blend_weight_pow = blend_weight_pow
        self.epsilon = 1e-6


        self.body_lbsw = get_body_lbsw(self.body_model)
        self.get_body_part_center()

        self.gen_grid_pts()

        os.makedirs(join(PROJECT_DIR, 'visualization', 'optimized_lbsw'), exist_ok=True)
        os.makedirs(join(PROJECT_DIR, 'visualization', 'dist_based_lbsw'), exist_ok=True)


    def gen_grid_pts(self):
        xs = torch.linspace(-1, 1, self.grid_resl)
        ys = torch.linspace(-1, 1, self.grid_resl)
        zs = torch.linspace(-0.4, 0.4, int(self.grid_resl/2.5))

        x, y, z = torch.meshgrid(xs, ys, zs)
        grid = torch.cat([y.reshape(-1, 1), x.reshape(-1, 1), z.reshape(-1, 1)], dim=-1)
        self.grid = grid # [N_pts, 3]
        return grid
    
    
    def get_body_part_center(self):
        '''
        body parts use argmax to get which bone they are max skinned to
        then compute their centroid
        '''

        self.num_bones = self.body_lbsw.shape[1]

        vert_bone_corr = torch.argmax(self.body_lbsw, dim=1).squeeze(0)

        centers = []
        per_bone_verts = {}
        per_bone_vert_lbsw = {}
        for i in range(self.num_bones):
            vert_idx = torch.where(vert_bone_corr==i)[0] # verts that are max skinned to bone i
            verts = self.cano_v[vert_idx]
            vert_lbsw = self.body_lbsw[...,vert_idx]

            centroid = verts.mean(dim=0)
            centers.append(centroid)
            per_bone_verts[i] = verts[None,:].cuda()
            per_bone_vert_lbsw[i] = vert_lbsw

        self.per_bone_verts = per_bone_verts
        self.per_bone_vert_lbsw = per_bone_vert_lbsw

        self.bone_centers = torch.stack(centers, dim=0).unsqueeze(0).float() # [num_bones, 3]
    

    def get_nearest_vert_lbsw(self):
        pass
    
    def barycentric_interpolation(self, val, coords):
        """
        :param val: verts x 3 x d input matrix
        :param coords: verts x 3 barycentric weights array
        :return: verts x d weighted matrix
        """
        t = val * coords[..., np.newaxis]
        ret = t.sum(axis=1)
        return ret

    def nearest_neighbor_diffusion(self, query_pts=None, return_nearest_dists=False):
        '''
        assumes the body mesh in cano space
        partially borrowed from LoopReg, 
        https://github.com/bharat-b7/LoopReg/blob/main/spread_SMPL_function.py

        args:
            query_pts: np array, [N, 3]

        return_nearest_dists: return the dists from each query point to the nearest point on the smpl surface
        '''
        smpl_mesh = Mesh(self.cano_v.numpy(), self.f.numpy())

        closest_face, closest_points = smpl_mesh.closest_faces_and_points(query_pts.numpy())
        vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(closest_points, closest_face.astype('int32'))

        nn_grid_lbsw = self.barycentric_interpolation(self.body_lbsw.squeeze(0).transpose(1,0).numpy()[vert_ids], bary_coords)

        if return_nearest_dists:
            dists = np.linalg.norm(closest_points - query_pts.numpy(), axis=-1)
            return nn_grid_lbsw, dists
        return nn_grid_lbsw

    


    def get_blend_weights_by_dist(self, dists):
        '''
        compute the weights to blend the LBSW of a query points' k nearest bones by dists of the query point to these bones

        dists: [N_query_pts, num_dists]. num_dists is the dist from each query pt to a set of fixed points
        '''
        weights = dists.pow(-self.blend_weight_pow) / dists.pow(-self.blend_weight_pow).sum(1).unsqueeze(1)

        return weights

    def get_grid_lbsw_with_threshold_onehot(self, query_pts, K=4):
        '''
        query_pts: [1, n_pts, 3]

        For each grid point as query point, compute its distance to all bone centers.

        NOTE: it potentially has discrete behavior at threshold boundary
        
        If the nearest dist is smaller than self.thres, then skin this query point to the bone whose center
        is nearest to the query point.
        The 'skin' is done by finding the group of body verts that are skinned to that bone, and assign the lbsw of the 
        closest point from this group of verts to the query point.
        
        If the nearest dist is larger than self.thres, then skin the query point to K nearest bones. This also uses
        the closest vert lbsw assignment mentioned above. To combine the lbsw of multiple bones, use a specific distance function.

        '''
        from pytorch3d.ops import knn_points
        print('computing query points knn wrt bone centers...')
        knn_out = knn_points(query_pts.cuda(), self.bone_centers.cuda(), K=K)
        dists, idx = knn_out[0], knn_out[1] # both have shape [1, num_query_pts, num_bone_centers]
        
        # first init the grid lbsw with naive nearest neighbor
        print('init with nearest neighbor...')
        grid_lbsw = self.nearest_neighbor_diffusion(self.grid)
        grid_lbsw = torch.tensor(grid_lbsw).float()
        nn_lbsw = grid_lbsw.clone()

        # then for those above threshold, blend K nearest bone's lbsw
        query_idx_above_thres = torch.where(dists[0,:,0] > self.thres)[0]
        bone_idx_above_thres = idx[0, :][query_idx_above_thres]

        print('generating one hot lbsw...')
        one_hot_lbsw = torch.zeros([len(bone_idx_above_thres) * K, self.num_bones]).cuda()
        bone_idx_above_thres = bone_idx_above_thres.reshape(-1)
        # for i in range(len(one_hot_lbsw)):
        #     one_hot_lbsw[i][bone_idx_above_thres[i]] = 1.0
        one_hot_lbsw.scatter_(dim=1, index=bone_idx_above_thres.unsqueeze(1), src=torch.ones_like(bone_idx_above_thres).unsqueeze(1).float())
        one_hot_lbsw = one_hot_lbsw.reshape(-1, K, self.num_bones)

        print('computing blend weights...')
        knn_bone_dists = dists[0, query_idx_above_thres]
        blend_w = self.get_blend_weights_by_dist(knn_bone_dists)
        blended_lbsw = torch.einsum('ij, ijk->ik', blend_w, one_hot_lbsw)

        grid_lbsw[query_idx_above_thres, :] = blended_lbsw.cpu()

        return grid_lbsw
    

    def get_grid_lbsw_with_threshold(self, query_pts, K=4):
        '''
        query_pts: [1, n_pts, 3]

        For each grid point as query point, compute its distance to all bone centers.

        NOTE: it potentially has discrete behavior at threshold boundary
        
        If the nearest dist is smaller than self.thres, then skin this query point to the bone whose center
        is nearest to the query point.
        The 'skin' is done by finding the group of body verts that are skinned to that bone, and assign the lbsw of the 
        closest point from this group of verts to the query point.
        
        If the nearest dist is larger than self.thres, then skin the query point to K nearest bones. This also uses
        the closest vert lbsw assignment mentioned above. To combine the lbsw of multiple bones, use a specific distance function.

        '''
        from pytorch3d.ops import knn_points
        print('computing query points knn wrt bone centers...')
        knn_out = knn_points(query_pts.cuda(), self.bone_centers.cuda(), K=K)
        dists, idx = knn_out[0], knn_out[1] # both have shape [1, num_query_pts, num_bone_centers]
        
        # first init the grid lbsw with naive nearest neighbor
        print('init with nearest neighbor...')
        grid_lbsw = self.nearest_neighbor_diffusion(self.grid)
        grid_lbsw = torch.tensor(grid_lbsw).float()
        nn_lbsw = grid_lbsw.clone()

        # then for those above threshold, blend K nearest bone's lbsw
        query_idx_above_thres = torch.where(dists[0,:,0] > self.thres)[0]
        bone_idx_above_thres = idx[0, :][query_idx_above_thres]

        knn_bone_dists = dists[0, query_idx_above_thres]
        
        # for each query point, get the vert groups that belong to its K nearest bones
        print('generating knn bone vert lbsw for each query point...')
        lbsw_fused_above_thres = []
        for i in tqdm(range(len(bone_idx_above_thres))):

            bone_idx = bone_idx_above_thres[i]
            per_bone_verts_i = [self.per_bone_verts[i.item()] for i in bone_idx_above_thres[0].cpu()]
            per_bone_vert_lbsw_i = [self.per_bone_vert_lbsw[i.item()] for i in bone_idx_above_thres[0].cpu()]
            
            # for each vert group, find the vert that's closest to the query point, and get its lbsw
            K_chosen_vert_lbsw = []
            for j, vert_group in enumerate(per_bone_verts_i):
                knn_out = knn_points(query_pts[:,[query_idx_above_thres[i]],:].cuda(), vert_group, K=1)
                idx_in_vert_group = knn_out[1].squeeze().item()
                chosen_vert_lbsw = per_bone_vert_lbsw_i[j][..., idx_in_vert_group]
                chosen_vert_lbsw = chosen_vert_lbsw.unsqueeze(1) # [1, 1, 22]
                K_chosen_vert_lbsw.append(chosen_vert_lbsw)

            K_chosen_vert_lbsw = torch.cat(K_chosen_vert_lbsw, dim=1)
            K_dists = knn_bone_dists[i].unsqueeze(0).cpu()
            K_weights = self.get_blend_weights_by_dist(K_dists)
            lbsw_fused = torch.einsum('ijk, ij->ik', K_chosen_vert_lbsw, K_weights)
            lbsw_fused_above_thres.append(lbsw_fused)
        
        blended_lbsw = torch.cat(lbsw_fused_above_thres, dim=0)

        grid_lbsw[query_idx_above_thres, :] = blended_lbsw.cpu()

        return grid_lbsw

        

if __name__ == '__main__':
    pow = sys.argv[1]
    grid_resl = int(sys.argv[2])

    '''
    getting the lbs weight field of the grid points determined by the points' location to the bones.
    not used in the final paper; kept here fore completeness
    '''
    
    diffuser = DiffuseLBSW(grid_resl=int(grid_resl), blend_weight_pow=int(pow), create_grid=True)
    diffuser.get_body_part_center()
    centers = diffuser.bone_centers

    grid_lbsw_Kbone_dist = diffuser.get_grid_lbsw_with_threshold(diffuser.grid.float()[None, :])
    grid_to_save = grid_lbsw_Kbone_dist.reshape(grid_resl, grid_resl, -1, diffuser.num_bones)
    torch.save(grid_to_save, join(PROJECT_DIR, 'visualization', 'dist_based_lbsw', 'grid_lbsw_Kbone_dist_resl{}.pt'.format(grid_resl)))
    color = color_lbsw(grid_lbsw_Kbone_dist, mode='diffuse', shuffle_color=True)
    customized_export_ply(join(PROJECT_DIR, 'visualization', 'dist_based_lbsw', 'grid_lbsw_Kbone_dist_resl{}_pow{}.ply'.format(grid_resl, pow)), diffuser.grid.cpu(), v_c=color)

    diffuser = DiffuseLBSW(grid_resl=int(grid_resl), blend_weight_pow=int(pow))
    diffuser.get_body_part_center()
    centers = diffuser.bone_centers

    grid_lbsw_onehot_fused = diffuser.get_grid_lbsw_with_threshold_onehot(diffuser.grid.float()[None, :])
    grid_to_save = grid_lbsw_onehot_fused.reshape(grid_resl, grid_resl, -1, diffuser.num_bones)
    torch.save(grid_to_save, join(PROJECT_DIR, 'visualization', 'dist_based_lbsw', 'grid_lbsw_onehot_fused_resl{}.pt'.format(grid_resl)))
    color = color_lbsw(grid_lbsw_onehot_fused, mode='diffuse', shuffle_color=True)
    customized_export_ply(join(PROJECT_DIR, 'visualization', 'dist_based_lbsw', 'grid_lbsw_onehot_fused_resl{}_pow{}.ply'.format(grid_resl, pow)), diffuser.grid.cpu(), v_c=color)