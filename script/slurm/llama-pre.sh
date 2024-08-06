#!/bin/sh

#SBATCH -J llama-pre
#SBATCH -p A100-80GB
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

srun python model/pretrain_llama.py \
--architecture llama \
--wandb_mode online \
--train_batch_size 4 \
--eval_batch_size 4 \
--gen_batch_size 32 \
--epochs 10 \
--max_length 100 \
--run_id 1azi0wgu



