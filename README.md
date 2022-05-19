# Heterogeneous-Agent Mirror Learning: A Continuum of Solutions to Cooperative MARL
Code for paper Heterogeneous-Agent Mirror Learning: A Continuum of Solutions to Cooperative MARL, this repository develops *Heterogeneous Agent Advantage Actor-Critic (HAA2C)* and *Heterogeneous-Agent Deep Deterministic Policy Gradient (HADDPG)* algorithms on the bechmarks of SMAC and Multi-agent MUJOCO. 

## Installation
For installation of necessary dependency and environments, please refer to the README in corresponding directory for HAA2C/HADDPG.

## How to run
For HAA2C/MAA2C,
``` Bash
cd haa2c/scripts
./train_mujoco_haa2c.sh  # run with HAA2C on Multi-agent MuJoCo
./train_mujoco_maa2c_nonshare.sh  # run with MAA2C-Moshare on Multi-agent MuJoCo
./train_mujoco_maa2c_share.sh  # run with MAA2C-Share on Multi-agent MuJoCo
./train_smac_haa2c.sh  # run with HAA2C on SMAC
./train_smac_maa2c_nonshare.sh  # run with MAA2C-Moshare on SMAC
./train_smac_maa2c_share.sh  # run with MAA2C-Share on SMAC
```

For HADDPG/MADDPG,
``` Bash
cd haddpg/examples/mujoco
bash ./run_maddpg.sh Hopper-v2 maddpg 3 result # run with MADDPG on Hopper-v2
bash ./run_maddpg.sh Hopper-v2 haddpg 3 result # run with HADDPG on Hopper-v2
```
If you would like to change the configs of experiments, you could modify sh files.


