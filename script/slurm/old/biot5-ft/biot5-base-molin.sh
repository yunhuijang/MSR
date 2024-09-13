#!/bin/sh

#SBATCH -J ft-biobase-molin
#SBATCH -p A100-80GB
#SBATCH -q add_hpgpu
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

srun python model/one_stage_generator.py \
--architecture biot5-plus-base-mol-instructions-molecule \
--cot_mode_multiset None \
--wandb_mode online \
--train_batch_size 32 \
--eval_batch_size 32 \
--cot_mode func_simple-chain-aromatic-con_ring_name \
--epochs 250 \
--weight_decay 0 \
--learning_rate 1e-3 \
--warmup_ratio 0.1 \
--check_val_every_n_epoch 5 \
--lr_scheduler_type cosine \
--max_length 820 \
--model_id QizhiPei




