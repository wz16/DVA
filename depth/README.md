# Depth Estimation with VPD
## Acknowledgement
This code uses the visual perception with pre-trained diffusion (VPD) model from [Wenjiang Zhao](https://github.com/wl-zhao/VPD).

## Getting Started

1. Install the [mmcv-full](https://github.com/open-mmlab/mmcv) library and some required packages.

```bash
pip install openmim
mim install mmcv-full
pip install -r requirements.txt
```

2. Prepare NYUDepthV2 datasets following [GLPDepth](https://github.com/vinvino02/GLPDepth) and [BTS](https://github.com/cleinc/bts/tree/master).

```
mkdir nyu_depth_v2
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./nyu_depth_v2/official_splits/
```

Download sync.zip provided by the authors of BTS from this [url](https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view) and unzip in `./nyu_depth_v2` folder. 

Your dataset directory should be:

```
│nyu_depth_v2/
├──official_splits/
│  ├── test
│  ├── train
├──sync/
```

3. Download the pre-trained VPD model from [here](https://cloud.tsinghua.edu.cn/f/27354f47ba424bb3ad40/?dl=1) and put it in the `./checkpoints` folder.

The original VPD code recommends using 8 NVIDIA V100 GPUs to train the model with a total batch size of 24. 

```
bash train.sh <LOG_DIR>
```

For evaluation:
```
bash test.sh <CHECKPOINT_PATH>
```

## Uncertainty Estimation
To run training script and visualize the uncertainty result, run the following command:
```
bash uncertainty.sh
```
The parameters are specified as arguments in `uncertainty.py`.