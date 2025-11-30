#!/bin/bash
#SBATCH --job-name=Training-MNCDV3
#SBATCH --output=./out_logs/%x-%A.out
#SBATCH --time=2-0:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:4

#SBATCH --partition=haicu_a100
#SBATCH --account=haicu

# module load python/3.11
module load apptainer

nvidia-smi

# OMP_NUM_THREADS=4 apptainer exec --nv --bind /bigdata:/bigdata /bigdata/3dabc/containers/ROSI/MNCDV3 accelerate launch train.py

# OMP_NUM_THREADS=4 NCCL_DEBUG=INFO NCCL_NVLS_ENABLE=0 apptainer exec --nv --bind /bigdata:/bigdata /bigdata/3dabc/containers/ROSI/MNCDV3 accelerate launch train.py

OMP_NUM_THREADS=4 NCCL_NVLS_ENABLE=0 apptainer exec --nv --bind /bigdata:/bigdata /bigdata/3dabc/containers/ROSI/MNCDV3 accelerate launch train.py