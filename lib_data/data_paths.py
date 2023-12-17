import os
from os.path import join, dirname, realpath
from datetime import date

PROJECT_DIR = join(dirname(realpath(__file__)), '..')

class DataPaths():
    def __init__(self):
        # modify the paths if needed
        # self.cape_raw_scan_sampled_root = join(PROJECT_DIR, 'data', 'cape_scans')
        # self.cape_packed_root = join(PROJECT_DIR, 'data', 'cape')
        # self.cape_packed_extra_root = join(PROJECT_DIR, 'data', 'cape_extra_params')

        self.resynth_packed_root = join(PROJECT_DIR, 'data', 'resynth')
        self.resynth_packed_extra_root = join(PROJECT_DIR, 'data', 'resynth_extra_params')

        self.smpl_path = join(PROJECT_DIR, 'body_models') # modify this if needed

        self.logs_path = join(PROJECT_DIR, 'checkpoints')
        self.samples_path = join(PROJECT_DIR, 'results', 'saved_samples')

        self.render_root = join(PROJECT_DIR, 'results', 'rendered')

        self.data_roots = {
            # 'cape': {'packed': self.cape_packed_root, 'extra': self.cape_packed_extra_root, 'scans': self.cape_raw_scan_sampled_root},
            'resynth': {'packed': self.resynth_packed_root, 'extra': self.resynth_packed_extra_root, 'scans': None},
        }

        # self.cape_subjs = {
        #     "00032_shortlong": 'male',
        #     "00096_jerseyshort": 'male',
        #     "00096_longshort": 'male',
        #     "00096_shirtlong": 'male',
        #     "00096_shirtshort": 'male',
        #     "00096_shortlong": 'male',
        #     "00096_shortshort": 'male',
        #     "00215_jerseyshort": 'male',
        #     "00215_longshort": 'male',
        #     "00215_poloshort": 'male',
        #     "03375_shortlong": 'male',
        #     "03375_blazerlong": 'male',
        #     "03375_longlong": 'male',
        #     "03375_shortshort": 'male'
        # }

        self.resynth_plain_subjs = [
            'rp_aaron_posed_002',
            'rp_aaron_posed_018',
            'rp_alexandra_posed_002',
            'rp_beatrice_posed_005',
            'rp_carla_posed_006',
            'rp_cindy_posed_005',
            'rp_corey_posed_002',
            'rp_corey_posed_006',
            'rp_corey_posed_010',
            'rp_eric_posed_006',
            'rp_ethan_posed_015',
            'rp_fiona_posed_013',
            'rp_henry_posed_014',
            'rp_janna_posed_032'
        ]

        self.resynth_skirt_subjs = [
            'rp_anna_posed_001',
            'rp_beatrice_posed_025',
            'rp_emma_posed_029',
            'rp_christine_posed_027',
            'rp_celina_posed_005',
            'rp_felice_posed_004',
            'rp_janett_posed_025',
            # 4 newly added
            'rp_carla_posed_016',
            'rp_cindy_posed_020',
            'rp_debra_posed_014',
            'rp_fernanda_posed_028',
        ]

        self.resynth_loose_subjs = [
            'rp_alexandra_posed_006',
            'rp_carla_posed_004',
            'rp_eric_posed_035',
        ]

        self.resynth_subj_chosen = [ # used in SkiRT paper
            'rp_anna_posed_001',
            'rp_beatrice_posed_025',
            'rp_christine_posed_027',
            'rp_carla_posed_004',
            'rp_celina_posed_005',
            'rp_felice_posed_004',
            'rp_janett_posed_025',
            'rp_debra_posed_014',
            'rp_eric_posed_035',
            'rp_eric_posed_006', # henley + wrinkly pants
            'rp_alexandra_posed_006',
            'rp_aaron_posed_002',
            'rp_aaron_posed_018',

        ]

    def set_up_experiment_paths(self, exp_args, project_dir):

        self.project_dir = project_dir

        self.data_root = self.data_roots[exp_args.dataset_type.lower()]['packed']
        self.data_root_extra = self.data_roots[exp_args.dataset_type.lower()]['extra']
        self.scan_root = self.data_roots[exp_args.dataset_type.lower()]['scans']

        # paths to save results
        self.samples_dir_val_base = join(self.samples_path, exp_args.name, 'val')
        self.samples_dir_test_seen_base = join(self.samples_path, exp_args.name, 'test_seen')
        os.makedirs(self.samples_dir_test_seen_base, exist_ok=True)
        os.makedirs(self.samples_dir_val_base, exist_ok=True)
        
        self.log_dir = join(PROJECT_DIR,'tb_logs/{}/{}'.format(date.today().strftime('%m%d'), exp_args.name))
        self.ckpt_dir = join(self.logs_path, exp_args.name)
        os.makedirs(self.ckpt_dir, exist_ok=True)