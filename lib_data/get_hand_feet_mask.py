
from os.path import join, dirname, realpath
import sys
PROJECT_DIR = join(dirname(realpath(__file__)), '..')
sys.path.append(PROJECT_DIR)

import numpy as np
from psbody.mesh import Mesh

from lib.utils_io import customized_export_ply
from lib.utils_model import get_body_lbsw

mode = 'for_faces' #'for_query_pts':

if mode == 'for_faces':
    for body_model in ['smpl', 'smplx']:
        import smplx

        model = smplx.create(model_type=body_model)

        lbsw = get_body_lbsw(model).numpy()
        
        hand_feet_def= {
            'smpl': np.array([7, 8, 10, 11, 20, 21, 22, 23]),
            'smplx': np.array([7, 8, 10, 11, 20, 21]),
        }

        hand_feet_idx = hand_feet_def[body_model]

        max_skinned_idx = np.argmax(lbsw, axis=1).squeeze()

        result = []
        stat = 0
        for idx in hand_feet_idx:
            is_this_part = (max_skinned_idx == idx).astype(np.int32)
            result.append(is_this_part)
            stat += len(np.where(is_this_part)[0])
        result = np.stack(result, axis=0)

        vert_is_handfeet = result.sum(0).astype(bool)
        vert_id_handfeet = np.where(vert_is_handfeet)[0]

        face_id_handfeet = []

        face_is_handfeet = vert_is_handfeet[model.faces.reshape(-1)].reshape(-1, 3).astype(np.int32).sum(-1)
        face_is_handfeet = face_is_handfeet.astype(bool)
        face_idx_handfeet = np.where(face_is_handfeet)[0]

        face_def_handfeet = model.faces[face_is_handfeet]

        np.save(join(PROJECT_DIR, 'assets', 'face_idx_handfeet_{}.npy'.format(body_model)), face_idx_handfeet)
        np.save(join(PROJECT_DIR, 'assets', 'face_mask_handfeet_{}.npy'.format(body_model)), face_is_handfeet)



if mode == 'for_query_pts':
    for body_model in ['smpl', 'smplx']:

        lbsw = np.load(join(PROJECT_DIR, 'assets', '{}_lbsw_query_pt.npy'.format(body_model)))
        pcl = Mesh(filename=join(PROJECT_DIR, 'assets', 'query_locs_3d_{}_pop_grid_256.ply'.format(body_model))).v

        hand_feet_def= {
            'smpl': np.array([7, 8, 10, 11, 20, 21, 22, 23]),
            'smplx': np.array([7, 8, 10, 11, 20, 21]),
        }

        hand_feet_idx = hand_feet_def[body_model]

        max_skinned_idx = np.argmax(lbsw, axis=1)
        result = []
        stat = 0
        for idx in hand_feet_idx:
            is_this_part = (max_skinned_idx == idx).astype(np.int32)
            result.append(is_this_part)
            stat += len(np.where(is_this_part)[0])
        result = np.stack(result, axis=0)
        
        mask_not_handfeet = ~result.sum(0).astype(bool)
        idx_not_handfeet = np.where(mask_not_handfeet)[0]
        mask_not_handfeet = mask_not_handfeet.astype(np.float32)
        print(idx_not_handfeet.shape)

        color = np.ones_like(pcl)
        color[:,2] = 0.
        color *= 255
        color[idx_not_handfeet] = 0.

        np.save(join(PROJECT_DIR, 'assets', 'non_handfeet_pt_mask_{}.npy'.format(body_model)), mask_not_handfeet)
        customized_export_ply(join(PROJECT_DIR, 'visualization', 'handfeet_mask_{}.ply'.format(body_model)), pcl, v_c=color)
