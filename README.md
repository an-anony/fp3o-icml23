# FP3O: Enabling Proximal Policy Optimization on Cooperative Multi-Agent Tasks with Diverse Network Types
***This is an anonymous repository for double-blind review at ICML 2023.***

This repository contains the implementation of Full-Pipline PPO (FP3O) on the bechmarks of MAMuJoCo and SMAC.

## 1. Installation
### Dependencies
``` Bash
conda create -n env_name python=3.7.12
conda activate env_name
pip install -r requirements.txt
```

### MAMuJoCo
Following the instructios in https://github.com/openai/mujoco-py and https://github.com/schroederdewitt/multiagent_mujoco to setup a mujoco environment. In the end, remember to set the following environment variables:
``` Bash
LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco200/bin;
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```
### StarCraft II & SMAC
Run the script
``` Bash
bash install_sc2.sh
```
Or you can follow here: https://github.com/oxwhirl/smac.

## 5. Traning
You could run shell scripts provided. For example:
``` Bash
cd scripts
./train_mujoco.sh  # run with FP3O on Multi-agent MuJoCo
./train_smac.sh  # run with FP3O on StarCraft II
```

If you would like to change the configs of experiments, you could modify sh files or look for config files for more details. And you can change network types by modify 
* **algo="fp3o_sha",--share_policy**, which denote FP3O with full parameter sharing,
* **algo="fp3o_par",--share_policy,--partial_share** , which denote FP3O with partial parameter sharing,
* **algo="fp3o_sep"** , which denote FP3O with non-parameter sharing.