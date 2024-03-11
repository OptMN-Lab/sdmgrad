# SDMGrad

Official implementation of *Direction-oriented Multi-objective Learning: Simple and Provable Stochastic Algorithms*.

## Supervised Learning
The expriments are conducted on Cityscapes and NYU-v2 datasets, which can be downloaded from [MTAN](https://github.com/lorenmt/mtan). Following [Nash-MTL](https://github.com/AvivNavon/nash-mtl) and [FAMO](https://github.com/Cranial-XIX/FAMO), the implementation is based on the `MTL` library.

### Setup Environment
Create the environment:
```
conda create -n mtl python=3.9.7
conda activate mtl
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113
```
Then, install the repo:
```
https://github.com/OptMN-Lab/sdmgrad.git
cd sdmgrad
python -m pip install -e .
```

### Run experiment
The dataset by default should be put under `experiments/EXP_NAME/dataset/` folder where EXP_NAME is chosen from `nyuv2, cityscapes`. To run the sdmgrad experiment:
```
cd experiments/EXP_NAME
sh run.sh
```


## Reinforcement Learning
The experiments are conducted on [Meta-World](https://github.com/Farama-Foundation/Metaworld) benchmark. To run the experiments on `MT10` and `MT50` (the instructions below are partly borrowed from [CAGrad](https://github.com/Cranial-XIX/CAGrad)):

1. Create python3.6 virtual environment.
2. Install the [MTRL](https://github.com/facebookresearch/mtrl) codebase.
3. Install the [Meta-World](https://github.com/Farama-Foundation/Metaworld) environment with commit id `d9a75c451a15b0ba39d8b7a8b6d18d883b8655d8`.
4. Copy the `mtrl_files` folder to the `mtrl` folder in the installed mtrl repo, then 

```
cd PATH_TO_MTRL/mtrl_files/ && chmod +x mv.sh && ./mv.sh
```

5. Follow the `run.sh` to run the experiments.
