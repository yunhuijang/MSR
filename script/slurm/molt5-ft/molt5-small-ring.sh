#!/bin/sh

#SBATCH -J ft-small-ring
#SBATCH -p A100-40GB
#SBATCH --gres=gpu:4
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

srun python train.py \
--architecture molt5-small \
--cot_mode_multiset full \
--cot_mode_ring \
--wandb_mode online \
--train_batch_size 64 \
--eval_batch_size 64



