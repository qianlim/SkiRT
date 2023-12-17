## Neural Point-based Shape Modeling of Humans in Challenging Clothing (3DV 2022)

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2109.01137)

This repository contains the official PyTorch implementation of the 3DV 2022 paper:

**Neural Point-based Shape Modeling of Humans in Challenging Clothing** <br>
Qianli Ma, Jinlong Yang, Michael J. Black and Siyu Tang <br>[Paper](https://ps.is.mpg.de/uploads_file/attachment/attachment/695/SkiRT_main_paper.pdf) | [Supp](https://ps.is.mpg.de/uploads_file/attachment/attachment/696/SkiRT_supp.pdf) | [Project website](https://qianlim.github.io/SkiRT) 

![](teasers/SkiRT_teaser.png)



## Installation

This repository is based on our prior work, [POP](https://github.com/qianlim/POP) and [SCALE](https://github.com/qianlim/SCALE). If you have successfully run either one of them before, you can skip steps 1 and 2 below. The code has been tested with python 3.8 on Ubuntu 20.04 + CUDA 11.0.

1. First, in the folder of this repository, run the following commands to create a new virtual environment and install dependencies:

```bash
python3 -m venv $HOME/.virtualenvs/SkiRT
source $HOME/.virtualenvs/SkiRT/bin/activate
pip install -U pip setuptools
pip install -r requirements.txt
```

2. Install the Chamfer Distance package (MIT license, taken from [this implementation](https://github.com/krrish94/chamferdist/tree/97051583f6fe72d5d4a855696dbfda0ea9b73a6a)). The compilation is verified to be successful under CUDA 11.0.

```bash
cd chamferdist
python setup.py install
cd ..
```

3. Install the `psbody.mesh` mesh processing library. Simply download the appropriate python wheels from their [Releases site](https://github.com/MPI-IS/mesh/releases) and simply install `pip install <downloaded_wheel_file>.whl`.
4. Download our pre-processed body model related assets for training SkiRT, such as the barycentric coordinates and face indices of the UV query points on the SMPL/SMPL-X body meshes. 

```bash
wget https://keeper.mpdl.mpg.de/f/9fc3ed8add544b4b9d8a/?dl=1 -O assets.zip && unzip assets.zip -d assets && rm assets.zip
```

You are now good to go with the next steps! All the commands below are assumed to be run from the `SkiRT` repository folder, within the virtual environment created above. 



## Download data

- Create folders for data and SMPL-X body model: `mkdir body_models data`.
- Download [SMPL-X body model](https://smpl-x.is.tue.mpg.de/) and place the model `.pkl` files under `body_models/smplx/`.
- Download the packed data [ReSynth dataset](https://pop.is.tue.mpg.de/).

  - After a simple registration, go to the "Download" tab, section "Option 2" (Download Data for Each Subject") there. Choose the subject(s) of interest, download the "packed packed npz files", unzip respectively to the `data/resynth/` folder. 
  - On the same download page, section "Pre-processed Attributes for SkiRT (3DV 2022)", download our pre-processed extra attributes of your selected subject(s) and unzip to the `data/resynth_extra_params/` folder. 
- The data file organization will look like this:

```
SkiRT
├── body_models
│   ├── smplx
│   │   │── SMPLX_FEMALE.pkl
│   │   │── SMPLX_MALE.pkl
│   │   ├── SMPLX_NEUTRAL.pkl
├── data
│   ├── resynth
│   │   ├── rp_anna_posed_001
│   │   │   ├── test
│   │   │   │   ├── <per-frame npz files>
│   │   │   ├── train
│   │   │   │   ├── <per-frame npz files>
│   │   ├── rp_beatrice_posed_025
│   │   ├── ...
│   ├── resynth_extra_params
│   │   ├── rp_anna_posed_001
│   │   │   ├── test
│   │   │   │   ├── <per-frame npz files>
│   │   │   ├── train
│   │   │   │   ├── <per-frame npz files>
│   │   ├── rp_beatrice_posed_025
│   │   ├── ...
```



## Training and Inference

### **Overview**

Training SkiRT involves training a coarse stage for pose-independent, coarse "template" of clothed body, followed by a fine stage that predicts the LBS weights and pose-dependent geometry.

Below we take the ReSynth dataset, subject `rp_beatrice_posed_025`, as an example, assuming the data are  in their corresponding folders under `data/`.

### **Train the coarse stage** 

```bash
python main.py --name experiment_rp_beatrice_posed_025 --config configs/config_coarse_stage.yaml --outfit_name rp_beatrice_posed_025 --mode train
```

- This command will start training for a pose-*in*dependent, coarse geometry of the clothed body represented by a point set (paper Sec. 4.1, Eq. 4).
- On an RTX6000 GPU this takes 0.15 min/epoch for a ReSynth subject, and ca. 20 minutes till convergence (150 epochs). 
- After training, the script will automatically evaluate it on the unsee poses of the same subject, and produce the learned coarse clothed body geometry under `results/saved_samples/experiment_rp_beatrice_posed_025/test_seen/coarse_stage_256/rp_beatrice_posed_025`, to be used in training the fine stage below, including:
  - `<frame_name>_pred.ply`: posed, coarse clothed body shape (not used, just for visualization).
  - `<frame_name>_pred_cano.ply`: predicted clothing on a T-posed, subject's personalized shaped body. This will be the base where the fine shape is added on the fine stage, i.e. $\hat{x}^{\textrm{coarse}}_i$ in paper Fig. 4.
  - `<frame_name>_pred_cano_mean.ply`: predicted clothing on a T-posed, SMPL-X mean-shaped default template mesh, represented as a point set. The points will be used to query the pre-diffused LBS weight field, which serves as an initialization of the LBS weight learning in the fine stage (see paper Sec. 4.1, last paragraph).

### Train the fine shape stage

```bash
python main.py --name experiment_rp_beatrice_posed_025 --config configs/config_fine_stage.yaml --outfit_name rp_beatrice_posed_025 --mode train
```

- This stage predicts a pose-dependent clothing offset field on top of the coarse shape learned above. The model is also trained to predict the LBS weights of the clothed body (paper Sec. 4.1, Eq. 5).
- Make sure the `--name` is the same as the coarse stage. In this example we use `experiment_rp_beatrice_posed_025` for both stages.
- On an RTX6000 GPU it takes ca. 0.8 min/epoch for a ReSynth subject, and ca. 2 hours till convergence (150 epochs). 
- The code will save the loss curves in the TensorBoard logs under `tb_logs/<date>/experiment_rp_beatrice_posed_025`.
- An example from the validation set at every N (default: 20) epochs will be saved at `results/saved_samples/experiment_rp_beatrice_posed_025/val`.
- After training, the script will automatically evaluate on the test split of the subject. The results will be saved in `results/saved_samples/experiment_rp_beatrice_posed_025/test_seen/fine_stage_256/rp_beatrice_posed_025`.

### Notes

- Each trained model (experiment) should be given a unique `name` by passing the `--name` flag to the code. The name is used for reading the seen/unseen outfit configurations in the config files and for the I/O of checkpoints, evaluation results, TensorBoard logs, etc. Make sure the coarse and fine stages have the same name.
- All the model arguments can either be passed by appending flags to the command line (see commands above as an example), or specifying them in the config yaml file. It's also possible to combine both; in case of a conflict, the flags passed in the command line will override those specified in the yaml.



## Misc
### Pre-diffusing the LBS weight field
- In `assets/pre_diffused_lbsw` we provide the optimized LBS weight field. We obtain them by propagating the body surface LBS weights into the 3D space, and performing an optimization for spatial smoothness.  This optimized field serves as an initialization to the LBS weight field learning in the fine stage (paper Sec. 4.1, last paragraph). 

- The code for this are provided in `lbs_grid_diffuse` just for information. To reproduce our pre-diffused LBS weight grid, run:

  ```bash
  python lbs_grid_diffuse/optim_lbsw.py
  ```

  The results will be saved to ` 'visualization/optimized_lbsw/`.

- A visualization of the colored LBS field is [available for download](https://keeper.mpdl.mpg.de/f/c0cea602cd1b45ba9e32/?dl=1) as .ply files (21M). The code for coloring the LBS weights (as shown in the teaser figure) are provided in `lib/utils_vis.py`.



## License

The scripts of this repository are licensed under the [MIT License](./LICENSE). The SMPL/SMPL-X body related files in the `assets/` folder are subject to the license of the [SMPL model](https://smpl.is.tue.mpg.de/) / [SMPL-X model](https://smpl-x.is.tue.mpg.de/). The Chamfer Distance implementation is subject to its [original license](./chamferdist/LICENSE).



## Citations

If you find the codes of this work or the associated ReSynth dataset helpful to your research, please consider citing:

```bibtex
@inproceedings{SkiRT:3DV:2022,
  title = {Neural Point-based Shape Modeling of Humans in Challenging Clothing},
  author = {Ma, Qianli and Yang, Jinlong and Black, Michael J. and Tang, Siyu},
  booktitle = {International Conference on 3D Vision (3DV)},
  pages = {679--689},
  month = sep,
  year = {2022}
}
```
