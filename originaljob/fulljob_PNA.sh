#!/usr/bin/env bash 

## Name of your SLURM job
#SBATCH --job-name=ResGatedFixDist

#SBATCH --output=outputs/fortest.out
#SBATCH --error=outputs/fortest.out

## Set a time limit for the job (in this case 1 day)
#SBATCH --time=48:00:00

## How much memory to request.Remove it on the oracle slurm cluster
##SBATCH --mem=100GB

## Partition to use,
#SBATCH --partition=student 

# CPU per task
## SBATCH --cpus-per-task=24

## Compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
set -e

# The below env variables can eventually help setting up your workload.
echo "SLURM_JOB_UID=$SLURM_JOB_UID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "********************************************************************"
echo "Starting sshd in Slurm as user"
echo "Environment information:"
echo "Date:" $(date)
echo "Allocated node:" $(hostname)
echo "Node IP:" $(ip a show eth0 | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1)
echo "Path:" $(pwd)
echo "Port:" $PORT
echo "********************************************************************"

# In a SLURM job, you CANNOT use `conda activate` and instead MUST use:
cd
source OliEnv/bin/activate


cd $HOME/MasterProject

# srun python file.py Architecture Weights? Layers? Skip? Testing? Selfloop? Distance? dataset ? seeds? Model_Type? Exp_number ? (Width ?)

srun python toyDatasetV1.py torchPNA NW 1 True both True 6 ZINC 1 GNN 15 32
srun python toyDatasetV1.py torchPNA NW 2 True both True 6 ZINC 1 GNN 15 32
srun python toyDatasetV1.py torchPNA NW 4 True both True 6 ZINC 1 GNN 15 32
srun python toyDatasetV1.py torchPNA NW 5 True both True 6 ZINC 10 GNN 15 32
srun python toyDatasetV1.py torchPNA NW 6 True both True 6 ZINC 10 GNN 15 32
srun python toyDatasetV1.py torchPNA NW 7 True both True 6 ZINC 10 GNN 15 32
srun python toyDatasetV1.py torchPNA NW 8 True both True 6 ZINC 10 GNN 15 32
srun python toyDatasetV1.py torchPNA NW 9 True both True 6 ZINC 10 GNN 15 32
srun python toyDatasetV1.py torchPNA NW 10 True both True 6 ZINC 10 GNN 15 32
srun python toyDatasetV1.py torchPNA NW 12 True both True 6 ZINC 10 GNN 15 32
# A dummy and useless `sleep` to give you time to see your job with `squeue`.
sleep 20s