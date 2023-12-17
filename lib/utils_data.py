from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
import numpy as np

def remove_faces(v, f, face_indices_to_remove, vn=None, vc=None):
    '''
    remove the faces of specified indices from a mesh.
    Adapted from psbody.mesh package.
    all inputs are np.array
    v,vn, vc are verts, vert normals, vert colors, shape=[V, 3]
    '''
    def arr_replace(arr_in, lookup_dict):
        arr_out = arr_in.copy()
        for k, v in lookup_dict.items():
            arr_out[arr_in == k] = v
        return arr_out
    num_v_orig = v.shape[0]
    f_new = np.delete(f, face_indices_to_remove, 0)
    v2keep = np.unique(f_new)
    v = v[v2keep]
    f = arr_replace(f_new, dict((v, i) for i, v in enumerate(v2keep)))

    if vn is not None and vn.shape[0] == num_v_orig:
        vn = vn[v2keep]
    if vc is not None and vc.shape[0] == num_v_orig:
        vc = vc[v2keep]

    return v, f, vn, vc

def point2mesh_dist(meshes: Meshes, pcls: Pointclouds, option='face2point'):
    '''
    mesh face to point distance, adapted from pytorch3d
    '''
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")

    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()
    
    if option == 'face2point':
        # point to face distance: shape (P,)
        dists, idxs = _C.face_point_dist_forward(
                points, points_first_idx, tris, tris_first_idx, max_points
        )
    
    else:
        dists, idxs = _C.point_face_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_points
        )

    return dists, idxs

    