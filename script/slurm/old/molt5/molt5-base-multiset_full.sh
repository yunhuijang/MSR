#!/bin/sh

#SBATCH -J molt5-base-multi_full
#SBATCH -p A5000
#SBATCH --gres=gpu:1
#SBATCH -o sbatch_log/%x.out

cd $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

date

nvidia-smi

srun python test.py \
--architecture molt5-base \
--cot_mode_multiset full

