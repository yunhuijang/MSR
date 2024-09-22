#!/bin/sh

#SBATCH -J t2m-reason-molt5-small
#SBATCH --exclude=n76,n56,n54,n52
#SBATCH -p 3090
#SBATCH --gres=gpu:6
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

srun python model/reasoning_generator.py \
--architecture molt5-small \
--cot_mode multiset_formula-chain-aromatic-con_ring_name-func_simple-chiral-weight-name \
--wandb_mode online \
--train_batch_size 32 \
--eval_batch_size 32 \
--epochs 250 \
--model_id laituan245 \
--max_length 820 \
--generation_mode \
--max_new_tokens 256 \
--weight_decay 0 \
--learning_rate 1e-3 \
--warmup_ratio 0.1 \
--lr_scheduler_type cosine \
--check_val_every_n_epoch 40




