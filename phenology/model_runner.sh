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

cd $HOME/SpongyMothIPM/phenology

source "/usr/projects/hpcsoft/common/x86_64/anaconda/2023.03-python-3.10/etc/profile.d/conda.sh"
conda activate model_fitting

srun --exclusive -n 1 --output=output_%j_0 python -u fit_models_v2.py --device=cuda -H 100 -W 100 -n 2000 -e 2000 &
srun --exclusive -n 1 --output=output_%j_1 python -u fit_models_v2.py --device=cuda -H 100 -W 100 -n 1000 -e 2000 &
srun --exclusive -n 1 --output=output_%j_2 python -u fit_models_v2.py --device=cuda -H 100 -W 100 -n 1500 -e 2000 &
srun --exclusive -n 1 --output=output_%j_3 python -u fit_models_v2.py --device=cuda -H 100 -W 100 -n 400 -e 2000 &
wait
