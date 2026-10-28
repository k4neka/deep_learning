#!/bin/bash -l
# CPU-only alternative (if the GPU queue is long). Slower per step than the GPU job.
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=06:00:00
#SBATCH --job-name=fire-ppo-cpu
#SBATCH --output=logs-%j.out

module purge
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl
source $VSC_DATA/dl-project/venv-fire/bin/activate
cd $VSC_DATA/dl-project/wildfire_kit_v4

python train_ppo.py --steps 2000000 --n-envs 32 --subproc  \
    --maps rolling_hills,mountain_valley,dense_forest,farmland --test-maps lake_shore,urban_interface \
    --difficulty easy,medium --test-difficulty medium --out runs/ppo_run
python render.py --policy ppo --model runs/ppo_run/model.zip --drones 3 --map lake_shore --out runs/ppo_run/ppo3_unseen_map.mp4
