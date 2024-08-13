# Code for paper: "Bridging MRI Cross-Modality Synthesis and Multi-Contrast Super-Resolution by Fine-Grained Difference Learning"

### Usage

#### 1. preprocess the data

#### Example of Data Folder Structure
```plaintext
ixi/
    ├───train/
    │   ├───t1/
    │   │   ├───HR/
    │   │   │   ├───001.png
    │   │   ├───LR4x/
    │   │   └───...
    │   ├───t2/

```
#### 2. train cms module
modify the data path in the config file
```commandline
python train.py --config ixi_cms.yaml
```
#### 3. train mcsr module
modify the data path and ckpt path in the config file
```commandline
python train.py --config ixi_sr.yaml
```

### Acknowledgement
Thanks for the code sharing of [Restormer](https://github.com/swz30/Restormer), [SynBoost](https://github.com/giandbt/synboost.git), this code is based on them.