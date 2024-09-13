#!/bin/sh

#SBATCH -J biot5p-base-mol2text
#SBATCH -p A6000
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

srun python evaluate_pretrained_model.py \
--architecture biot5-plus-base \
--split test \
--model_id QizhiPei \
--finetune_task=-text2mol \
--wandb_mode online 

