#!/bin/bash -l
# Generic CPU job: runs any script of this folder with the arguments given after the job file.
#   sbatch --job-name=NAME jobs/exp_cpu.slurm train_ppo.py --steps 1500000 ... --out runs/NAME
#   sbatch --job-name=NAME jobs/exp_cpu.slurm evaluate_models.py --runs runs/a runs/b ... --out runs/eval_x
# The gpu_a100 queue was days long in Nov 2026; this small CNN trains fine on CPU.
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=logs-%x-%j.out

module purge
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl
source $VSC_DATA/dl-project/venv-fire/bin/activate
cd $SLURM_SUBMIT_DIR

echo "job $SLURM_JOB_NAME on $(hostname), $(date)"
echo "python $@"
python "$@"
