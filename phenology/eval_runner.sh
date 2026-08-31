#!/bin/bash
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=output_%j
#SBATCH --array=1-176

cd $HOME/SpongyMothIPM/phenology

source "/usr/projects/hpcsoft/common/x86_64/anaconda/2023.03-python-3.10/etc/profile.d/conda.sh"
conda activate model_fitting

srun --exclusive -n 1 --output=eval_output_%j_0 python -u evaluate_models.py --device=cuda -H 200 -W 200 -n $((4*(SLURM_ARRAY_TASK_ID-1))) &
srun --exclusive -n 1 --output=eval_output_%j_1 python -u evaluate_models.py --device=cuda -H 200 -W 200 -n $((4*(SLURM_ARRAY_TASK_ID-1)+1)) &
srun --exclusive -n 1 --output=eval_output_%j_2 python -u evaluate_models.py --device=cuda -H 200 -W 200 -n $((4*(SLURM_ARRAY_TASK_ID-1)+2)) &
srun --exclusive -n 1 --output=eval_output_%j_3 python -u evaluate_models.py --device=cuda -H 200 -W 200 -n $((4*(SLURM_ARRAY_TASK_ID-1)+3)) &
wait