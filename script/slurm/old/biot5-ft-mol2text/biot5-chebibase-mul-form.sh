#!/bin/sh

#SBATCH -J ft-biobaseche-mulform-m2t-genmode
#SBATCH -p A6000
#SBATCH --gres=gpu:8
#SBATCH -o sbatch_log/%x.out
#SBATCH -n 8

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

srun python model/one_stage_generator_mol2text.py \
--architecture biot5-plus-base-chebi20 \
--cot_mode multiset_formula-func_simple-chain-aromatic-con_ring_name \
--wandb_mode online \
--train_batch_size 16 \
--eval_batch_size 16 \
--epochs 250 \
--model_id QizhiPei \
--weight_decay 0 \
--learning_rate 5e-4 \
--warmup_ratio 0.1 \
--check_val_every_n_epoch 5 \
--lr_scheduler_type linear \
--max_length 820 \
--generation_mode \
--max_new_tokens 512




