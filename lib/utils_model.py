import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

def get_body_lbsw(body_model_loaded, hires=False):
    '''
    ! partially taken from scanimate

    body_model_loaded: the loaded smpl/smplx body model instance

    gt_lbsw: [1, num_verts, num_joints]

    returns: skinning weights of each vertex from the given body model, [1, n_joints, n_verts]
    '''
    # for now, all the hand, face joints are combined with body joints for smplx
    body_model_name = type(body_model_loaded).__name__.lower()
    max_joint_id_clo_related = 23 if body_model_name == 'smpl' else 21 # doesn't include root joint
    
    if not hires:
        gt_lbs_smpl = body_model_loaded.lbs_weights
        out_lbsw = gt_lbs_smpl.clone().detach()[:,:max_joint_id_clo_related+1]
    else:
        gt_lbs_smpl = np.load('assets/lbsw_{}_hires_verts.npy'.format(body_model_name))
        out_lbsw = torch.tensor(gt_lbs_smpl)[:, :max_joint_id_clo_related+1].clone()

    root_idx = body_model_loaded.parents.cpu().numpy()
    idx_list = list(range(root_idx.shape[0]))
    
    for i in range(root_idx.shape[0]):
        if i > max_joint_id_clo_related:
            root = idx_list[root_idx[i]]
            out_lbsw[:,root] += gt_lbs_smpl[:,i]
            idx_list[i] = root
    out_lbsw = out_lbsw[None].permute(0,2,1).float()
    return out_lbsw


def gen_transf_mtx_full_uv(verts, faces):
    '''
    given a positional uv map, for each of its pixel, get the matrix that transforms the prediction from local to global coordinates
    The local coordinates are defined by the posed body mesh (consists of vertcs and faces)

    :param verts: [batch, Nverts, 3]
    :param faces: [uv_size, uv_size, 3], uv_size =e.g. 32
    
    :return: [batch, uv_size, uv_size, 3,3], per example a map of 3x3 rot matrices for local->global transform

    NOTE: local coords are NOT cartesian! uu an vv axis are edges of the triangle,
          not perpendicular (more like barycentric coords)
    '''
    tris = verts[:, faces] # [batch, uv_size, uv_size, 3, 3]
    v1, v2, v3 = tris[:, :, :, 0, :], tris[:, :, :, 1, :], tris[:, :, :, 2, :]
    uu = v2 - v1 # u axis of local coords is the first edge, [batch, uv_size, uv_size, 3]
    vv = v3 - v1 # v axis, second edge
    ww_raw = torch.cross(uu, vv, dim=-1)
    ww = F.normalize(ww_raw, p=2, dim=-1) # unit triangle normal as w axis
    ww_norm = (torch.norm(uu, dim=-1).mean(-1).mean(-1) + torch.norm(vv, dim=-1).mean(-1).mean(-1)) / 2.
    ww = ww*ww_norm.view(len(ww_norm),1,1,1)
    
    # shape of transf_mtx will be [batch, uv_size, uv_size, 3, 3], where the last two dim is like:
    #  |   |   |
    #[ uu  vv  ww]
    #  |   |   |
    # for local to global, say coord in the local coord system is (r,s,t)
    # then the coord in world system should be r*uu + s*vv+ t*ww
    # so the uu, vv, ww should be colum vectors of the local->global transf mtx
    # so when stack, always stack along dim -1 (i.e. column)
    transf_mtx = torch.stack([uu, vv, ww], dim=-1)

    return transf_mtx


def gen_transf_mtx_from_vtransf(vtransf, bary_coords, faces, scaling=1.0):
    '''
    interpolate the local -> global coord transormation given such transformations defined on 
    the body verts (pre-computed) and barycentric coordinates of the query points from the uv map.

    Note: The output of this function, i.e. the transformation matrix of each point, is not a pure rotation matrix (SO3).
    
    args:
        vtransf: [batch, #verts, 3, 3] # per-vertex rotation matrix
        bary_coords: [uv_size, uv_size, 3] # barycentric coordinates of each query point (pixel) on the query uv map 
        faces: [uv_size, uv_size, 3] # the vert id of the 3 vertices of the triangle where each uv pixel locates

    returns: 
        [batch, uv_size, uv_size, 3, 3], transformation matrix for points on the uv surface
    '''
    #  
    vtransf_by_tris = vtransf[:, faces] # shape will be [batch, uvsize, uvsize, 3, 3, 3], where the the last 2 dims being the transf (pure rotation) matrices, the other "3" are 3 points of each triangle
    transf_mtx_uv_pts = torch.einsum('bpqijk,pqi->bpqjk', vtransf_by_tris, bary_coords) # [batch, uvsize, uvsize, 3, 3], last 2 dims are the rotation matix
    transf_mtx_uv_pts *= scaling
    
    return transf_mtx_uv_pts


class SampleSquarePoints():
    def __init__(self, npoints=16, min_val=0, max_val=1, device='cuda', include_end=True):
        super(SampleSquarePoints, self).__init__()
        self.npoints = npoints
        self.device = device
        self.min_val = min_val # -1 or 0
        self.max_val = max_val # -1 or 0
        self.include_end = include_end

    def sample_regular_points(self, N=None):
        steps = int(self.npoints ** 0.5) if N is None else int(N ** 0.5)
        if self.include_end:
            linspace = torch.linspace(self.min_val, self.max_val, steps=steps) # [0,1]
        else:
            linspace = torch.linspace(self.min_val, self.max_val, steps=steps+1)[: steps] # [0,1)
        grid = torch.stack(torch.meshgrid([linspace, linspace]), -1).to(self.device) #[steps, steps, 2]
        grid = grid.view(-1,2).unsqueeze(0) #[B, N, 2]
        grid.requires_grad = True

        return grid

    def sample_random_points(self, N=None):
        npt = self.npoints if N is None else N
        shape = torch.Size((1, npt, 2))
        rand_grid = torch.Tensor(shape).float().to(self.device)
        rand_grid.data.uniform_(self.min_val, self.max_val)
        rand_grid.requires_grad = True #[B, N, 2]
        return rand_grid


class Embedder():
    '''
    Simple positional encoding, adapted from NeRF: https://github.com/bmild/nerf
    '''
    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    '''
    Helper function for positional encoding, adapted from NeRF: https://github.com/bmild/nerf
    '''
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class PositionalEncoding():
    def __init__(self, input_dims=2, num_freqs=10, include_input=False):
        super(PositionalEncoding,self).__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.input_dims = input_dims

    def create_embedding_fn(self):
        embed_fns = []
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        freq_bands = 2. ** torch.linspace(0, self.num_freqs-1, self.num_freqs)
        periodic_fns = [torch.sin, torch.cos]

        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq:p_fn(math.pi * x * freq))
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq:p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self,coords):
        '''
        use periodic positional encoding to transform cartesian positions to higher dimension
        :param coords: [N, 3]
        :return: [N, 3*2*num_freqs], where 2 comes from that for each frequency there's a sin() and cos()
        '''
        return torch.cat([fn(coords) for fn in self.embed_fns], dim=-1)


def normalize_uv(uv):
    '''
    normalize uv coords from range [0,1] to range [-1,1]
    '''
    return uv * 2. - 1.


def uv_to_grid(uv_idx_map, resolution):
    '''
    uv_idx_map: shape=[batch, N_uvcoords, 2], ranging between 0-1
    this function basically reshapes the uv_idx_map and shift its value range to (-1, 1) (required by F.gridsample)
    the sqaure of resolution = N_uvcoords
    '''
    bs = uv_idx_map.shape[0]
    grid = uv_idx_map.reshape(bs, resolution, resolution, 2) * 2 - 1.
    grid = grid.transpose(1,2)
    return grid



def pix_coord_convert(pix_coords, resl=256, mode='center_to_corner'):
    '''
    if each pixel is treated as a point, and the min pix coord=0, max=1, then the N'th pixel's x-coord will be (1/resolution-1)*N+ 0
    if each pixel is treated as a square, its coord is the center of the square. Then the N's pixel's x-coord will be (1/resolution)*N + 1/(2*resolution)

    this func provides conversion between the two conventions
    '''
    if mode == 'center_to_corner':
        pix_coords = (pix_coords - 1 / (2*resl)) * resl / (resl - 1)
    else:
        pix_coords = pix_coords * (resl - 1) / resl + 1 / (2*resl)
    return pix_coords
                

def flip_pix_coords_up_down(pix_coords, mode='cv_to_gl'):
    ''''
    our positional maps uses OpenGL convention: x right, y up
    but the smpl uv template uses the opencv (or numpy matrix) convention: x (first dim) down, y (second dim) right

    this function converts between the two

    pix_coords: [B, N_pts, 2], value range between 0 and 1
    '''
    if mode == 'cv_to_gl':
        dim1 = pix_coords[...,:1]
        dim2 = pix_coords[..., 1:]
        dim2 = 1 - dim2
        return torch.cat([dim1, dim2], -1)
    else:
        return NotImplementedError



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    from torch.nn import init
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_mlp_siren(m):
    from torch.nn import init
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        d_in = m.weight.data.size()[1]
        init.uniform_(m.weight.data, -math.sqrt(6/d_in), math.sqrt(6/d_in))
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))

class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def interp_features_dev(faces, vert_features, bary_coords, sample_face_idxs, interp_same_bary_idx=True):
    '''
    assume sampling the same set of N bary + selected face id on different meshes


    vert_features: [B, C, Nv]
    bary_coords: [B, N, 3]
    sample_face_idxs = [N]
    faces: [num_faces, 3], num_faces are from the original mesh
    '''
    if interp_same_bary_idx:
        B, feat_dim, _ = vert_features.shape 
        n_points = bary_coords.shape[1]
        if len(sample_face_idxs.shape) > 1:
            sample_face_idxs = sample_face_idxs.squeeze(0)

        interped_features = torch.zeros([B, feat_dim, n_points])

        w0, w1, w2 = bary_coords[...,0], bary_coords[...,1], bary_coords[...,2]

        vert_features = vert_features.permute([0,2,1]) # [B, n_verts, feat_dim]

        tris = vert_features[:, faces, :] # [B, n_faces, 3, feat_dim], vert features arranged in triangles

        tris = tris.view(-1, 3, feat_dim)

        feat_v0, feat_v1, feat_v2 = tris[:, 0, :], tris[:, 1, :], tris[:, 2, :] # [B*n_faces, feat_dim]

        # get feature on the selected (sampled) triangles' verts
        feat_v0_sel = feat_v0.reshape(B, -1, feat_dim)[:, sample_face_idxs, :] # [B, n_points, feat_dim]
        feat_v1_sel = feat_v1.reshape(B, -1, feat_dim)[:, sample_face_idxs, :]
        feat_v2_sel = feat_v2.reshape(B, -1, feat_dim)[:, sample_face_idxs, :]

        interped_features = w0[:, :, None] * feat_v0_sel + w1[:, :, None] * feat_v1_sel + w2[:, :, None] * feat_v2_sel

    else:
        B, C, Nv = vert_features.shape
        faces_list = faces.expand(B, -1, -1)
        faces_packed = torch.cat([i*Nv + x for i, x in enumerate(faces_list)], dim=0) # a tensor of faces, i.e. *vertex indices*
        
        vert_features = vert_features.permute(0,2,1).reshape(-1, C) #[B, Nv, C]

        num_meshes = len(vert_features)
        # Initialize samples tensor with fill value 0 for empty meshes.
        num_samples = bary_coords.shape[1]
        samples = torch.zeros((num_meshes, num_samples, 3), device=vert_features.device)
        
        # Get the vertex coordinates of the sampled faces.
        face_verts = vert_features[faces_packed]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        # Randomly generate barycentric coords.
        w0, w1, w2 = bary_coords[...,0], bary_coords[...,1], bary_coords[...,2]

        # Use the barycentric coords to get a point on each sampled face.
        a = v0[sample_face_idxs]  # (N, num_samples, 3) # note sample_face_idxs are a tensor of *face* indices, not to confuse with vertex indices above
        b = v1[sample_face_idxs]
        c = v2[sample_face_idxs]
        interped_features = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

    return interped_features

def index_custom(feat, uv):
    '''
    copy-pasted from scanimate
    Author: Shunsuke Saito

    args:
        feat: (B, C, H, W)
        uv: (B, 2, N)
    return:
        (B, C, N)
    '''
    device = feat.device
    B, C, H, W = feat.size()
    _, _, N = uv.size()
    
    x, y = uv[:,0], uv[:,1]
    x = (W-1.0) * (0.5 * x.contiguous().view(-1) + 0.5)
    y = (H-1.0) * (0.5 * y.contiguous().view(-1) + 0.5)

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    max_x = W - 1
    max_y = H - 1

    x0_clamp = torch.clamp(x0, 0, max_x)
    x1_clamp = torch.clamp(x1, 0, max_x)
    y0_clamp = torch.clamp(y0, 0, max_y)
    y1_clamp = torch.clamp(y1, 0, max_y)

    dim2 = W
    dim1 = W * H

    base = (dim1 * torch.arange(B).int()).view(B, 1).expand(B, N).contiguous().view(-1).to(device)

    base_y0 = base + y0_clamp * dim2
    base_y1 = base + y1_clamp * dim2

    idx_y0_x0 = base_y0 + x0_clamp
    idx_y0_x1 = base_y0 + x1_clamp
    idx_y1_x0 = base_y1 + x0_clamp
    idx_y1_x1 = base_y1 + x1_clamp

    # (B,C,H,W) -> (B,H,W,C)
    im_flat = feat.permute(0,2,3,1).contiguous().view(-1, C)
    i_y0_x0 = torch.gather(im_flat, 0, idx_y0_x0.unsqueeze(1).expand(-1,C).long())
    i_y0_x1 = torch.gather(im_flat, 0, idx_y0_x1.unsqueeze(1).expand(-1,C).long())
    i_y1_x0 = torch.gather(im_flat, 0, idx_y1_x0.unsqueeze(1).expand(-1,C).long())
    i_y1_x1 = torch.gather(im_flat, 0, idx_y1_x1.unsqueeze(1).expand(-1,C).long())
    
    # Check the out-of-boundary case.
    x0_valid = (x0 <= max_x) & (x0 >= 0)
    x1_valid = (x1 <= max_x) & (x1 >= 0)
    y0_valid = (y0 <= max_y) & (y0 >= 0)
    y1_valid = (y1 <= max_y) & (y1 >= 0)

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    w_y0_x0 = ((x1 - x) * (y1 - y) * (x1_valid * y1_valid).float()).unsqueeze(1)
    w_y0_x1 = ((x - x0) * (y1 - y) * (x0_valid * y1_valid).float()).unsqueeze(1)
    w_y1_x0 = ((x1 - x) * (y - y0) * (x1_valid * y0_valid).float()).unsqueeze(1)
    w_y1_x1 = ((x - x0) * (y - y0) * (x0_valid * y0_valid).float()).unsqueeze(1)

    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1 # (B, N, C)

    return output.view(B, N, C).permute(0,2,1).contiguous()


def mesh_grid_points(H, W, N_subsample):
    # partially borrowed from AtlasNet code
    # works for a grid of N*N points
    assert H == W
    pqroot = int(N_subsample ** 0.5) # number of points on each `edge' of the pq square sample
    grain = pqroot * H - 1

    faces = []

    for i in range(1,int(grain + 1)):
        for j in range(0,(int(grain + 1)-1)):
            faces.append([ j+(grain+1)*i, j+(grain+1)*i + 1, j+(grain+1)*(i-1)])
    for i in range(0,(int((grain+1))-1)):
        for j in range(1,int((grain+1))):
            faces.append([j+(grain+1)*i, j+(grain+1)*i - 1, j+(grain+1)*(i+1)])

    return faces

def mesh_grid_points_only_valid_pts(H, W, N_subsample):
    # mesh within each uv island
    assert H == W
    pqroot = int(N_subsample ** 0.5) # number of points on each `edge' of the pq square sample
    grain = pqroot * H - 1

    faces = []

    for i in range(1,int(grain + 1)):
        for j in range(0,(int(grain + 1)-1)):
            faces.append([ j+(grain+1)*i, j+(grain+1)*i + 1, j+(grain+1)*(i-1)])
    for i in range(0,(int((grain+1))-1)):
        for j in range(1,int((grain+1))):
            faces.append([j+(grain+1)*i, j+(grain+1)*i - 1, j+(grain+1)*(i+1)])

    return faces


def calc_bary_coord(p, a, b, c):
    '''
    reference: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    p: query point, [3]
    a,b,c: vertices of the triangle, each has shape [3]
    return: 1x3 array of the barycentric coords
    '''
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return np.array([u,v,w])

def calc_bary_coord_batch(p, a, b, c):
    '''
    reference: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    p: query point, Nx3
    a,b,c: vertices of the triangle, Nx3 each
    return: u, v, w: 3 components of the barycentric coords, Nx3 each
    '''
    v0 = b - a
    v1 = c - a
    v2 = p - a
    
    def batch_dot(vec1, vec2):
        # [N, 3], [N, 3] --> [N], batch-wise dot product
        return torch.einsum('ij, ij->i', vec1, vec2)

    d00 = batch_dot(v0, v0)
    d01 = batch_dot(v0, v1)
    d11 = batch_dot(v1, v1)
    d20 = batch_dot(v2, v0)
    d21 = batch_dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return torch.stack([u,v,w], dim=-1).float()

def get_bary_coords(body_verts, faces_chosen, query_points):
    '''
    given the query points and each point's face_id on the body mesh, 
    calculate the barycentric coordinate of each point in their corresponding triangles
    
    body_verts: [V, 3]
    query_points: [N, 3]
    faces_chosen: [N, 3], the faces on the body model that the query_pts are correspond to

    returns:
    bary_coords: [N, 3]
    '''

    vt = torch.tensor(body_verts).float()
    faces_chosen = torch.tensor(faces_chosen).long()
    query_points = torch.tensor(query_points).float()

    tris = vt[faces_chosen]
    v1, v2, v3 = tris[:, 0, :], tris[:, 1, :], tris[:, 2, :]

    v1, v2, v3 = list(map(lambda x: x.reshape(-1, 3), [v1, v2, v3]))

    bary_coords = calc_bary_coord_batch(query_points, v1, v2, v3)
    return bary_coords



def get_body_vert_bary(body_mesh, device='cuda', debug=False):
    '''
    body_mesh: a psbody.Mesh object of the body
    returns: bary coords [V, 3] and the face indices [V] that the verts belong to 
    '''
    from lib.utils_model import interp_features_dev
    import torch

    verts = body_mesh.v
    closest_face, _ = body_mesh.closest_faces_and_points(verts)
    verts_faces, verts_bary = body_mesh.barycentric_coordinates_for_points(verts, closest_face) # each smpl vert, choose a face it locates on, and the corresp. bary coords (should all be [1 0 0] or similar)

    verts_bary = (verts_bary > 0.5).astype(np.float32)

    # to verify
    if debug:
        out = interp_features_dev(
            faces=torch.tensor(body_mesh.f.astype(np.int32)).long(),
            vert_features=torch.tensor(verts).transpose(1,0).unsqueeze(0),
            bary_cords=torch.tensor(verts_bary).unsqueeze(0),
            sample_face_idx = torch.tensor(closest_face.astype(np.int32)).squeeze().long()
            )

        diff = out.squeeze().numpy() - verts
        print(np.abs(diff).max())

    return torch.tensor(verts_bary).unsqueeze(0).to(device), torch.tensor(closest_face.astype(np.int32)).long().to(device)


class FeatureVolume(nn.Module):
    '''
    adapted from NGLOD paper: https://github.com/nv-tlabs/nglod/blob/8441414146d785c926782a30f544bb6f700f94ad/sdf-net/lib/models/OctreeSDF.py
    '''
    def __init__(self, vol_values):
        super().__init__()
        self.fm = vol_values.float() # [B, C, H, W, D], D = depth dim, i.e. the front-back dimension of body
        self.fm = self.fm.permute([0,1,4,2,3]) # [B, C, D, H, W]

    def forward(self, x):
        '''
        x: query points, no batch dim, [N_pt, 3]
        '''
        N = x.shape[0]
        if x.shape[1] == 3:
            sample_coords = x.reshape(1, N, 1, 1, 3) # [N, 1, 1, 3]    
            sample = F.grid_sample(self.fm, sample_coords, 
                                   align_corners=True, mode='bilinear', padding_mode='border')[0,:,:,0,0].transpose(0,1)
        else:
            sample_coords = x.reshape(1, N, x.shape[1], 1, 3) # [N, 1, 1, 3]    
            sample = F.grid_sample(self.fm, sample_coords, 
                                   align_corners=True, mode='bilinear', padding_mode='border')[0,:,:,:,0].permute([1,2,0])
        return sample
