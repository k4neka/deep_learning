#!/bin/bash -l
# v4: train on ALL four fire engines (a different engine per episode = domain randomisation across
# published simulators), then test every policy on each engine separately ("cross_simulator").
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --job-name=fire-ppo-allfire
#SBATCH --output=logs-%j.out

module purge
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl
source $VSC_DATA/dl-project/venv-fire/bin/activate
cd $VSC_DATA/dl-project/wildfire_kit_v4

nvidia-smi
python train_ppo.py --steps 3000000 --n-envs 32 --subproc --device cuda \
    --maps rolling_hills,mountain_valley,dense_forest,farmland --test-maps lake_shore,urban_interface \
    --difficulty easy,medium --test-difficulty medium \
    --fire-models native,pyrorl,haksar,rothermel --test-fire-models native,pyrorl,haksar,rothermel \
    --out runs/ppo_allfire
python render.py --policy ppo --model runs/ppo_allfire/model.zip --drones 3 --map lake_shore \
    --fire-model rothermel --out runs/ppo_allfire/ppo3_unseen_map_rothermel.mp4
