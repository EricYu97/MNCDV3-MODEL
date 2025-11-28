#!/bin/bash
#SBATCH --job-name=Training-MNCDV3
#SBATCH --output=%x-%A.out
#SBATCH --time=2-0:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:4

#SBATCH --partition=gpu-h100


# module load python/3.11
module load apptainer

OMP_NUM_THREADS=4 apptainer exec --nv --bind /bigdata:/bigdata /bigdata/3dabc/MNCD/container/MMCV/ python train.py