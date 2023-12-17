import os
from os.path import join, dirname
import glob

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from psbody.mesh import Mesh

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

class CloDataSet(Dataset):
    def __init__(self, data_root=None, data_root_extra=None, split='train', body_model='smpl', dataset_type='cape',
                 sample_spacing=1, query_posmap_size=256, inp_posmap_size=128, scan_npoints=-1,
                 pose_input='posmap', use_raw_scan=False, scan_root=None,
                 dataset_subset_portion=1.0, outfits={}):

        self.dataset_type = dataset_type
        self.data_root = data_root
        self.data_root_extra = data_root_extra

        self.scan_root = scan_root

        self.data_dirs = {outfit: join(data_root, outfit, split) for outfit in outfits.keys()} # will be sth like "./data/packed/cape/00032_shortlong/train"
        self.data_dirs_extra = {outfit: join(data_root_extra, outfit, split) for outfit in outfits.keys()} # will be sth like "./data/packed/cape/00032_shortlong/train"
        self.dataset_subset_portion = dataset_subset_portion # randomly subsample a number of data from each clothing type (using all data from all outfits will be too much)
        self.query_posmap_size = query_posmap_size
        self.inp_posmap_size = inp_posmap_size
        self.use_raw_scan = use_raw_scan

        self.num_clo_types = len(self.data_dirs)

        self.pose_input = pose_input # can be 'posmap' (will use standard UNet); 'points': will use convOcc encoder; 'verts' (will use meshconv)

        self.split = split
        self.query_posmap_size = query_posmap_size
        self.spacing = sample_spacing
        self.scan_npoints = scan_npoints
        self.f = np.load(join(SCRIPT_DIR, '..', 'assets', '{}_faces.npy'.format(body_model)))
        self.clo_label_def = outfits

        # get the valid pixel's index on the positional map
        self.valid_idxs = {}
        for resl in [128, 256]:
            uv_mask_faceid = np.load(join(SCRIPT_DIR, '../assets', 'uv_masks', 'uv_mask{}_with_faceid_{}.npy'.format(resl, body_model))).reshape(resl, resl)
            self.valid_idxs[resl] = (uv_mask_faceid!= -1).ravel()

        self.query_posmap, self.posmap_meanshape, self.scan_n, self.scan_pc = [], [], [], []
        self.scan_name, self.body_verts, self.body_pose, self.clo_labels =  [], [], [], []
        self.posed_subj_v, self.posed_mean_v, self.cano_subj_bs_v =  [], [], []
        self.vtransf, self.jT = [], []
        self.raw_scans = []

        self._init_dataset()
        self.data_size = int(len(self.query_posmap))

        print('Data loaded, in total {} {} examples.\n'.format(self.data_size, self.split))

    def _init_dataset(self):
        print('Loading {} data...'.format(self.split))

        flist_all = []
        subj_id_all = []

        for outfit_id, (outfit, outfit_datadir) in enumerate(self.data_dirs.items()):
            flist = sorted(glob.glob(join(outfit_datadir, '*.npz')))[::self.spacing]
            print('Loading {}, {} examples..'.format(outfit, len(flist)))
            flist_all = flist_all + flist
            subj_id_all = subj_id_all + [outfit.split('_')[0]] * len(flist)

        if self.dataset_subset_portion < 1.0:
            import random
            random.shuffle(flist_all)
            num_total = len(flist_all)
            num_chosen = int(self.dataset_subset_portion*num_total)
            flist_all = flist_all[:num_chosen]
            print('Total examples: {}, now only randomly sample {} from them...'.format(num_total, num_chosen))
        

        for idx, fn in enumerate(tqdm(flist_all)):
            fn_extra = fn.replace(self.data_root, self.data_root_extra)

            dd = np.load(fn)
            dd_extra = np.load(fn_extra)
            self.body_pose.append(torch.tensor(dd_extra['body_pose']).float())

            clo_type = dirname(fn).split('/')[-2] # e.g. longlong 
            clo_label = self.clo_label_def[clo_type] # the numerical label of the type in the lookup table (outfit_labels.json)
            self.clo_labels.append(torch.tensor(clo_label).long())
            self.query_posmap.append(torch.tensor(dd['posmap{}'.format(self.query_posmap_size)]).float().permute([2,0,1]))

            # for historical reasons in the packed data the key is called "posmap_canonical"
            # it actually stands for the positional map of the *posed, mean body shape* of SMPL/SMPLX (see POP paper Sec 3.2)
            # which corresponds to the inp_posmap_ms in the train and inference code 
            # if the key is not available, simply use each subject's personalized body shape.

            # HACK! make proper posmap canonical 256.
            if 'posmap_canonical{}'.format(self.inp_posmap_size) not in dd.files:
                self.posmap_meanshape.append(torch.tensor(dd['posmap{}'.format(self.inp_posmap_size)]).float().permute([2,0,1]))
            else:
                self.posmap_meanshape.append(torch.tensor(dd['posmap_canonical{}'.format(self.inp_posmap_size)]).float().permute([2,0,1]))

            # in the packed files the 'scan_name' field doensn't contain subj id, need to append it
            scan_name_loaded = str(dd['scan_name'])
            scan_name = scan_name_loaded if scan_name_loaded.startswith('0') else '{}_{}'.format(subj_id_all[idx], scan_name_loaded)
            self.scan_name.append(scan_name)

            if not self.use_raw_scan:
                self.scan_pc.append(torch.tensor(dd['scan_pc']).float())
                self.scan_n.append(torch.tensor(dd['scan_n']).float())
            else:
                fn_scan = fn.replace(self.data_root, self.scan_root).replace('.npz', '.ply')
                scan = Mesh(filename=fn_scan)
                self.scan_pc.append(torch.tensor(scan.v).float())
                self.scan_n.append(torch.tensor(scan.vn).float())

            self.posed_subj_v.append(torch.tensor(dd_extra['subj_body_v']).float())
            self.posed_mean_v.append(torch.tensor(dd_extra['mean_body_v']).float())
            self.cano_subj_bs_v.append(torch.tensor(dd_extra['subj_body_v_cano_bs']).float())
            self.jT.append(torch.tensor(dd_extra['jT_pa']).float())
            
            vtransf = torch.tensor(dd['vtransf']).float()
            if vtransf.shape[-1] == 4:
                vtransf = vtransf[:, :3, :3]
            self.vtransf.append(vtransf)

  
        self.clo_labels = torch.stack(self.clo_labels)


    def get_pts_from_posmap_single(self, posmap):
        resl = posmap.shape[-1] # [3, resl, resl]
        valid_idx = self.valid_idxs[resl]
        pts = posmap.permute(1,2,0).reshape(-1, 3)[valid_idx]
        return pts

    def get_pts_from_posmap_batch(self, posmap_batch):
        B, C, resl, _ = posmap_batch.shape # [B, 3, resl, resl]
        valid_idx = self.valid_idxs[resl]
        pts = posmap_batch.view(B, C, -1)[..., valid_idx].permute(0,2,1).contiguous()
        return pts

    def __getitem__(self, index):
        query_posmap = self.query_posmap[index]

        # in mean SMPL/ SMPLX body shape but in the same pose as the original subject
        if self.pose_input=='posmap':
            pose_inp_tensor = self.posmap_meanshape[index]
        elif self.pose_input == 'points':
            pose_inp_tensor = self.normalized_minimal_posed[index]
        
        scan_name = self.scan_name[index]
        posed_subj_v = self.posed_subj_v[index]
        posed_mean_v = self.posed_mean_v[index] 
        cano_subj_bs_v = self.cano_subj_bs_v[index] 

        clo_label = self.clo_labels[index]

        scan_n = self.scan_n[index]
        scan_pc = self.scan_pc[index]

        body_pose = self.body_pose[index]

        vtransf = self.vtransf[index]
        jT = self.jT[index]

        if self.scan_npoints != -1: 
            selected_idx = torch.randperm(len(scan_n))[:self.scan_npoints]
            scan_pc = scan_pc[selected_idx, :]
            scan_n = scan_n[selected_idx, :]

        return query_posmap, pose_inp_tensor, scan_n, scan_pc, vtransf, jT, scan_name, body_pose, posed_subj_v, posed_mean_v, cano_subj_bs_v, clo_label

    def __len__(self):
        return self.data_size


class IndexData(Dataset):
    def __init__(self, idx):
        self.idx = idx

    def __getitem__(self, index):
        return self.idx[index]

    def __len__(self):
        return len(self.idx)


class CustomSampler(torch.utils.data.RandomSampler):
    '''
    https://cloud.tencent.com/developer/article/1728103
    '''

    def __init__(self, data, shuffle=True):
        self.data = data
        self.uniq_clo_labels = list(self.data.clo_label_def.values())
        self.shuffle = shuffle

    def __iter__(self):
        indices = []
        for n in self.uniq_clo_labels:
            index = torch.where(self.data.clo_labels == n)[0]
            # shuffle indices within each clothing type 
            rand_perm = torch.randperm(len(index))
            index = index[rand_perm]
            indices.append(index)
        indices = torch.cat(indices, dim=0)
        return iter(indices)
    
    def gen_index_dict(self):
        indices = {}
        for n in self.uniq_clo_labels:
            index = torch.where(self.data.clo_labels == n)[0]
            if self.shuffle:
                rand_perm = torch.randperm(len(index))
                index = index[rand_perm]
            indices[n] = index
        return indices

    def __len__(self):
        return len(self.data)


class CustomBatchSampler_orig:
    '''
    https://cloud.tencent.com/developer/article/1728103
    '''
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.uniq_clo_labels = self.sampler.uniq_clo_labels

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)

        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            if (
                i < len(sampler_list) - 1
                and self.sampler.data.clo_labels[idx]
                != self.sampler.data.clo_labels[sampler_list[i + 1]]
            ):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
                else:
                    batch = []
            i += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


            

class CustomBatchSampler:
    '''
    https://cloud.tencent.com/developer/article/1728103
    '''
    def __init__(self, sampler, batch_size, drop_last=False, shuffle=True):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.uniq_clo_labels = self.sampler.uniq_clo_labels
        self.shuffle = shuffle

    def get_total_num_batches(self, num_examples):
        if num_examples % self.batch_size == 0:
            return num_examples // self.batch_size
        else:
            return num_examples // self.batch_size + 1 

    def __iter__(self):
        batch = []
        i = 0
        
        sampler_dict = self.sampler.gen_index_dict()
        num_batch_per_type = {k: self.get_total_num_batches(len(v)) for k, v in sampler_dict.items()}
        clo_type_sample_list = torch.cat([torch.tensor(k).expand(v) for k, v in num_batch_per_type.items()])
        randperm = torch.randperm(len(clo_type_sample_list))
        if self.shuffle:
            clo_type_sample_list = clo_type_sample_list[randperm]
        batch_counter = {k:0 for k in self.uniq_clo_labels}

        for j, clo_type in enumerate(clo_type_sample_list):
            clo_type = clo_type.item()

            start_idx = batch_counter[clo_type] * self.batch_size

            if start_idx + self.batch_size > len(sampler_dict[clo_type]):
                batch = sampler_dict[clo_type][start_idx: ]
            else:
                end_idx = start_idx + self.batch_size
                batch = sampler_dict[clo_type][start_idx: end_idx]
            batch_counter[clo_type] += 1
            yield batch
        batch_counter = {k:0 for k in self.uniq_clo_labels}

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

if __name__ == '__main__':
    '''
    quickly verify the data storage structure and dataset construction
    '''
    from torch.utils.data import DataLoader

    PROJECT_DIR = ''
    data_root = join(PROJECT_DIR, '../POP_dev', 'data', 'resynth_orig', 'packed')
    data_root_extra = join(PROJECT_DIR, '../POP_dev', 'data', 'resynth', 'packed')

    dataset_config = {
                    'dataset_type': 'resynth',
                    'body_model': 'smplx',
                    'data_root': data_root,
                    'data_root_extra': data_root_extra,
                    'query_posmap_size':256, 
                    'inp_posmap_size': 128,
                    'pose_input': 'posmap',
                    }

    train_set = CloDataSet(split='train', outfits={'rp_anna_posed_001': 0, 'rp_eric_posed_035': 1, 'rp_beatrice_posed_025': 2},
                        sample_spacing=10,
                        dataset_subset_portion=1.0, **dataset_config)
    basic_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)

    custom_sampler = CustomSampler(train_set, shuffle=True)
    batch_sampler = CustomBatchSampler(custom_sampler, 5, False, shuffle=True)
    custom_loader = DataLoader(train_set, num_workers=4, batch_sampler=batch_sampler)
    
    all_data = []
    for epoch in range(2):
        for idx, data in enumerate(custom_loader):
            print(idx, data[-3])
            all_data += list(data[6])
        all_data = []