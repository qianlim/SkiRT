from typing import Tuple, Union
import torch

def adjust_loss_weights(init_weight, current_epoch, mode='decay', start=400, every=20, rise_rate=1.05, decay_rate=0.9):
    # decay or rise the loss weights according to the given policy and current epoch
    # mode: decay, rise or binary

    if mode != 'binary':
        if current_epoch < start:
            if mode == 'rise':
                weight = init_weight * 1e-6 # use a very small weight for the normal loss in the beginning until the chamfer dist stabalizes
            else:
                weight = init_weight
        else:
            if every == 0:
                weight = init_weight # don't rise, keep const
            else:
                if mode == 'rise':
                    weight = init_weight * (rise_rate ** ((current_epoch - start) // every))
                else:
                    weight = init_weight * (decay_rate ** ((current_epoch - start) // every))

    return weight


def compute_knn_dist(pcl, n_ngbr=5, amplt=2, num_sigmas_thres=1, max_power=10, exponential=False, use_thres=False):
    '''
    pcl: [B, N, 3]
    '''
    from pytorch3d.io import load_ply
    from pytorch3d.structures import Meshes, Pointclouds
    from pytorch3d.ops import knn_points, knn_gather

    import torch

    knn_out = knn_points(pcl, pcl, K=n_ngbr)
    dists, idx = knn_out[0], knn_out[1]

    dists_avg = torch.sqrt(dists)[...,1:].mean(-1) # dists returned by pytorch3d knn are squared; [1:] removes the nearest ngbr (a point itself)
    
    mm = dists_avg.mean(-1)[:, None]
    std = dists_avg.std(-1)[:, None]

    per_point_power = (dists_avg - mm) / std # for a skirt, per_point_power max=20.xx, min=-2.xx, mean~0, median=-0.04
    per_point_power = torch.clamp(per_point_power, max=max_power*amplt)

    if exponential:
        per_point_weight = 2 ** (per_point_power / amplt) # hope that the bigger knn nbghr, the larger weights, here uses exponential
    else:
        per_point_weight = per_point_power / amplt

    if use_thres:
        # CAUTION if use thres, then in a batch diff examples will have diff num of points
        thres = dists_avg.mean() + dists_avg.std() * num_sigmas_thres
        per_point_weight[dists_avg < thres] = 0.

    return per_point_weight


def get_adaptive_point_loss_weights(pcl, base_weight, n_ngbr=5, amplt=2, num_sigmas_thres=1, max_power=10, binary=False, use_thres=False):
    '''
    computing Adaptive loss weight as described in SkiRT paper Sec. 4.3. 

    pcl: [B, N, 3]

    binary: if True, when a point's avg knn is below threshold, its loss weigths are set as a normal value; else it's set to zero.
            if False, it'll be set as a exponentially decaying value
    '''
    from pytorch3d.io import load_ply
    from pytorch3d.structures import Meshes, Pointclouds
    from pytorch3d.ops import knn_points, knn_gather

    import torch

    knn_out = knn_points(pcl, pcl, K=n_ngbr)
    dists, idx = knn_out[0], knn_out[1]

    dists_avg = torch.sqrt(dists)[...,1:].mean(-1) # dists returned by pytorch3d knn are squared; [1:] removes the nearest ngbr (a point itself)
    
    mm = dists_avg.mean(-1)[:, None]
    std = dists_avg.std(-1)[:, None]

    per_point_power = (dists_avg - mm) / std # for a skirt, per_point_power max=20.xx, min=-2.xx, mean~0, median=-0.04
    per_point_power = torch.clamp(per_point_power, max=max_power*amplt)

    if use_thres:
        thres = dists_avg.mean() + dists_avg.std() * num_sigmas_thres
        if binary:
            per_point_weight = torch.zeros_like(per_point_power).to(pcl.device)
            per_point_weight[dists_avg < thres] = base_weight.to(pcl.device)

        else:
            per_point_weight = 2 ** (-1 * per_point_power / amplt)
            per_point_weight[dists_avg < thres] = base_weight.to(pcl.device)
    
    return per_point_weight

    
def sample_points_from_meshes(
    meshes,
    num_samples: int = 10000,
    return_bary_coords: bool = True,
    return_normals: bool = False,
    return_textures: bool = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    ! Taken and adapted from pytorch3d 0.6.1 source
    ! added the option to return the barycentric coordinates
    !!! CAUTION: if there are more than 1 mesh in a batch, the returned sampled face indices will be the 'packed' version (similar to pytorch3d faces_packed)

    ==== below are original doc from pytorch3d
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.
        return_normals: If True, return normals for the sampled points.
        return_textures: If True, return textures for the sampled points.

    Returns:
        3-element tuple containing

        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
          For empty meshes the corresponding row in the normals array will
          be filled with 0.
        - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
          texture vector to each sampled point. Only returned if return_textures is True.
          For empty meshes the corresponding row in the textures array will
          be filled with 0.

        Note that in a future releases, we will replace the 3-element tuple output
        with a `Pointclouds` datastructure, as follows

        .. code-block:: python

            Pointclouds(samples, normals=normals, features=textures)
    """
    import sys

    import torch
    from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
    from pytorch3d.ops.packed_to_padded import packed_to_padded
    from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments
    from pytorch3d.ops.sample_points_from_meshes import _rand_barycentric_coords

    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    if return_textures and meshes.textures is None:
        raise ValueError("Meshes do not contain textures.")

    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(
            areas, mesh_to_face[meshes.valid], max_faces
        )  # (N, F)

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        sample_face_idxs = areas_padded.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)
        sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_valid_meshes, num_samples, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

    if return_normals:
        # Initialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are face normals computed from
        # the vertices of the face in which the sampled point lies.
        normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals[sample_face_idxs]
        normals[meshes.valid] = vert_normals

    if return_textures:
        # fragment data are of shape NxHxWxK. Here H=S, W=1 & K=1.
        pix_to_face = sample_face_idxs.view(len(meshes), num_samples, 1, 1)  # NxSx1x1
        bary = torch.stack((w0, w1, w2), dim=2).unsqueeze(2).unsqueeze(2)  # NxSx1x1x3
        # zbuf and dists are not used in `sample_textures` so we initialize them with dummy
        dummy = torch.zeros(
            (len(meshes), num_samples, 1, 1), device=meshes.device, dtype=torch.float32
        )  # NxSx1x1
        fragments = MeshFragments(
            pix_to_face=pix_to_face, zbuf=dummy, bary_coords=bary, dists=dummy
        )
        textures = meshes.sample_textures(fragments)  # NxSx1x1xC
        textures = textures[:, :, 0, 0, :]  # NxSxC

    # return
    # TODO(gkioxari) consider returning a Pointclouds instance [breaking]
    if return_normals and return_textures:
        # pyre-fixme[61]: `normals` may not be initialized here.
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, normals, textures
    # if return_normals:  # return_textures is False
    #     # pyre-fixme[61]: `normals` may not be initialized here.
    #     return samples, normals
    if return_textures:  # return_normals is False
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, textures
    if return_bary_coords:
        bary = torch.stack((w0, w1, w2), dim=2)
        if return_normals:
            return samples, normals, sample_face_idxs, bary
        else:
            return samples, sample_face_idxs, bary

        
    return samples




def sample_points_from_meshes_weighted(
    meshes,
    num_samples: int = 10000,
    weights: torch.Tensor = None,
    return_bary_coords: bool = True,
    return_normals: bool = False,
    return_textures: bool = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    ! Taken and adapted from pytorch3d 0.6.1 source
    ! added the option to return the barycentric coordinates

    ==== below are original doc from pytorch3d
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.
        return_normals: If True, return normals for the sampled points.
        return_textures: If True, return textures for the sampled points.

    Returns:
        3-element tuple containing

        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
          For empty meshes the corresponding row in the normals array will
          be filled with 0.
        - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
          texture vector to each sampled point. Only returned if return_textures is True.
          For empty meshes the corresponding row in the textures array will
          be filled with 0.

        Note that in a future releases, we will replace the 3-element tuple output
        with a `Pointclouds` datastructure, as follows

        .. code-block:: python

            Pointclouds(samples, normals=normals, features=textures)
    """
    import sys

    import torch
    from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
    from pytorch3d.ops.packed_to_padded import packed_to_padded
    from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments
    from pytorch3d.ops.sample_points_from_meshes import _rand_barycentric_coords

    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    if return_textures and meshes.textures is None:
        raise ValueError("Meshes do not contain textures.")

    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(
            areas, mesh_to_face[meshes.valid], max_faces
        )  # (N, F)
        if weights is not None:
            if len(weights.shape) == 2: # [B, num_faces], the weight per face differs across examples in a batch
                assert areas_padded.shape == weights.shape
                areas_padded_w = areas_padded * weights
            elif len(weights.shape) == 1: # [num_faces], same set of face weights for all examples in a batch
                assert areas_padded.shape[1] == len(weights)
                areas_padded_w = weights[None,:] * areas_padded
        else:
            areas_padded_w = areas_padded
        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        sample_face_idxs = areas_padded_w.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)
        sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)
    
    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_valid_meshes, num_samples, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

    if return_normals:
        # Initialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are face normals computed from
        # the vertices of the face in which the sampled point lies.
        normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals[sample_face_idxs]
        normals[meshes.valid] = vert_normals

    if return_textures:
        # fragment data are of shape NxHxWxK. Here H=S, W=1 & K=1.
        pix_to_face = sample_face_idxs.view(len(meshes), num_samples, 1, 1)  # NxSx1x1
        bary = torch.stack((w0, w1, w2), dim=2).unsqueeze(2).unsqueeze(2)  # NxSx1x1x3
        # zbuf and dists are not used in `sample_textures` so we initialize them with dummy
        dummy = torch.zeros(
            (len(meshes), num_samples, 1, 1), device=meshes.device, dtype=torch.float32
        )  # NxSx1x1
        fragments = MeshFragments(
            pix_to_face=pix_to_face, zbuf=dummy, bary_coords=bary, dists=dummy
        )
        textures = meshes.sample_textures(fragments)  # NxSx1x1xC
        textures = textures[:, :, 0, 0, :]  # NxSxC

    # return
    # TODO(gkioxari) consider returning a Pointclouds instance [breaking]
    if return_normals and return_textures:
        # pyre-fixme[61]: `normals` may not be initialized here.
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, normals, textures
    # if return_normals:  # return_textures is False
    #     # pyre-fixme[61]: `normals` may not be initialized here.
    #     return samples, normals
    if return_textures:  # return_normals is False
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, textures
    if return_bary_coords:
        bary = torch.stack((w0, w1, w2), dim=2)
        if return_normals:
            return samples, normals, sample_face_idxs, bary
        else:
            return samples, sample_face_idxs, bary

        
    return samples


def get_points_by_packed_bary_coords(
    meshes,
    sample_face_idxs,
    bary_coords, 
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    adapted from pytorch3d sample_points_from_meshes()

    takes in pytorch3d.structures.meshes object (containing verts and faces, assume same topology) 
    bary_coords: [B, N_sample, 3]
    sample_face_idxs:the direct output of the sample_points_from_meshes_weighted() function (note: these are not the face indices of the
                                     original meshes, but of the packed meshes in pytorch3d)
    """
    import sys

    import torch
    from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
    from pytorch3d.ops.packed_to_padded import packed_to_padded
    from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments
    from pytorch3d.ops.sample_points_from_meshes import _rand_barycentric_coords

    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    faces = meshes.faces_packed()
    num_meshes = len(meshes)
    
    # Initialize samples tensor with fill value 0 for empty meshes.
    num_samples = bary_coords.shape[1]
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
    
    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = bary_coords[...,0], bary_coords[...,1], bary_coords[...,2]
    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

    return samples

def pack_face_idx_by_batch(face_idx, batch_size, num_faces_body_model):
    '''
    face_idx: [B, N], N=number of faces (triangles). Each member is the index of a triangle (note: not the index of vertex)
    same as the faces_packed() in pytorch3d.meshes
    '''
    if len(face_idx) == 1: # when all examples are sampled in a same way (use the same set of face_id, so B=1):
        face_idx_packed = [face_idx + num_faces_body_model*i for i in range(batch_size)]
        face_idx_packed = torch.stack(face_idx_packed, dim=0).reshape(-1)
    else: # when all examples are sampled differently, i.e. face_idx are different in a batch
        face_idx_packed = [x + i*num_faces_body_model for i, x in enumerate(face_idx)]
        face_idx_packed = torch.stack(face_idx_packed, dim=0)
    return face_idx_packed


def unpack_face_idx_by_batch(face_idx, num_faces_body_model):
    '''
    face_idx: [B, N], N=number of faces (triangles). Each member is the index of a triangle (note: not the index of vertex)
    pack_face_idx_by_batch
    '''
    face_idx_unpacked = [x - i*num_faces_body_model for i, x in enumerate(face_idx)]
    face_idx_unpacked = torch.stack(face_idx_unpacked, dim=0)
    return face_idx_unpacked


def adaptive_sampling(pred_pts_posed, clo_model, query_face_idx_original, posed_subj_v=None, max_power=8, num_pt_adaptive=5000):
    '''
    adaptively sample new points according to triangle (face) area
    '''
    from pytorch3d.structures import Meshes

    bs = pred_pts_posed.shape[0]
    adapt_sample_weights_per_point = compute_knn_dist(pred_pts_posed, amplt=1.5, num_sigmas_thres=1.5, max_power=max_power,
                                                    exponential=True, use_thres=True) # per point

    weights_per_face = torch.zeros([bs, len(clo_model.faces)]).view(-1).cuda() # TODO
    num_pts_per_face = torch.zeros([bs, len(clo_model.faces)]).view(-1).cuda() 

    query_face_idx_packed = pack_face_idx_by_batch(query_face_idx_original, bs, len(clo_model.faces))

    weights_per_face.put_(query_face_idx_packed, adapt_sample_weights_per_point.view(-1), accumulate=True)
    num_pts_per_face.put_(query_face_idx_packed, 
                        torch.ones_like(adapt_sample_weights_per_point).cuda().view(-1), accumulate=True)
    num_pts_per_face[num_pts_per_face == 0] = 1.

    weights_per_face = weights_per_face / num_pts_per_face
    weights_per_face = weights_per_face.reshape(bs, len(clo_model.faces)).contiguous()

    # num_pt_adaptive = kwargs['num_pt_adaptve'] // (i+1)
    cano_meshes_to_sample = Meshes(verts=clo_model.cano_vt.expand(bs, -1, -1), faces=clo_model.faces.expand(bs, -1, -1))

    sampled_pts_cano, sampled_face_idx_new, sampled_bary_new = sample_points_from_meshes_weighted(meshes=cano_meshes_to_sample, 
                            weights=weights_per_face, num_samples=num_pt_adaptive, return_bary_coords=True)

    posed_meshes_to_sample = Meshes(verts=posed_subj_v, faces=clo_model.faces.expand(bs, -1, -1))
    sampled_pts_posed = get_points_by_packed_bary_coords(posed_meshes_to_sample, sampled_face_idx_new, sampled_bary_new)

    return sampled_pts_cano, sampled_pts_posed, sampled_face_idx_new, sampled_bary_new 



    
    