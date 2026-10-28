#!/bin/bash -l
# Recommended for v2: the network update is the bottleneck, so the GPU helps.
# 32 CPU cores run 32 copies of the simulator in parallel; the A100 does the learning.
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --job-name=fire-ppo-gpu
#SBATCH --output=logs-%j.out

module purge
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl
source $VSC_DATA/dl-project/venv-fire/bin/activate
cd $VSC_DATA/dl-project/wildfire_kit_v4

nvidia-smi
python train_ppo.py --steps 3000000 --n-envs 32 --subproc --device cuda \
    --maps rolling_hills,mountain_valley,dense_forest,farmland --test-maps lake_shore,urban_interface \
    --difficulty easy,medium --test-difficulty medium --out runs/ppo_run
python render.py --policy ppo --model runs/ppo_run/model.zip --drones 3 --map lake_shore --out runs/ppo_run/ppo3_unseen_map.mp4
