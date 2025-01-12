#!/bin/sh

#SBATCH -J eval-m2t-molt5-large
#SBATCH --exclude=n76,n56,n54,n52
#SBATCH -p A100-80GB
#SBATCH --gres=gpu:4
#SBATCH -o sbatch_log/%x.out
#SBATCH -q hpgpu

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
--architecture molt5-large \
--cot_mode multiset_formula-chain-aromatic-con_ring_name-func_simple-chiral \
--wandb_mode online \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 82 \
--model_id laituan245 \
--weight_decay 0 \
--learning_rate 6e-4 \
--warmup_ratio 0.1 \
--check_val_every_n_epoch 1 \
--lr_scheduler_type linear \
--max_length 820 \
--generation_mode \
--max_new_tokens 512 \
--dataset_name lm \
--run_id r24po3do \
--is_lm_eval