#!/bin/bash
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=????
#SBATCH --output=output%j

cd $HOME/SpongyMothIPM/phenology

conda activate model_fitting

srun --exact -n 1 -c 10 -G 1 --mem-per-cpu 10G python fit_models.py --device=cuda -h 1000 -w 1000 -n 0 -e 2000 &
srun --exact -n 1 -c 10 -G 1 --mem-per-cpu 10G python fit_models.py --device=cuda -h 1000 -w 1000 -n 1 -e 2000 &
srun --exact -n 1 -c 10 -G 1 --mem-per-cpu 10G python fit_models.py --device=cuda -h 1000 -w 1000 -n 2 -e 2000 &
srun --exact -n 1 -c 10 -G 1 --mem-per-cpu 10G python fit_models.py --device=cuda -h 1000 -w 1000 -n 3 -e 2000 &
wait