#!/bin/sh

#SBATCH -J t2m-answer-molt5-base-iter
#SBATCH --exclude=n76,n56,n54,n52
#SBATCH -p RTX6000ADA
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
--cot_mode multiset_formula-chain-aromatic-con_ring_name-func_simple-chiral \
--wandb_mode online \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 250 \
--model_id laituan245 \
--max_length 820 \
--generation_mode \
--max_new_tokens 512 \
--weight_decay 0.01 \
--learning_rate 0.00002 \
--warmup_ratio 0 \
--lr_scheduler_type linear \
--check_val_every_n_epoch 20 \
--is_iterative




