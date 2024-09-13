#!/bin/sh

#SBATCH -J answer-base-mul
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

srun python model/answer_generator.py \
--architecture molt5-base \
--cot_mode func_simple-chain-aromatic-con_ring_name \
--wandb_mode online \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 250 \
--max_length 512 \
--model_id laituan245 \
--max_length 820 \
--check_val_every_n_epoch 5




