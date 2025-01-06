#!/bin/sh

#SBATCH -J t2m-chemt5-base-reason
#SBATCH --exclude=n76,n56,n54,n52
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


srun python model/reasoning_generator.py \
--architecture multitask-text-and-chemistry-t5-base-standard \
--cot_mode multiset_formula-chain-aromatic-con_ring_name-func_simple-chiral \
--wandb_mode online \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 250 \
--model_id GT4SD \
--weight_decay 0 \
--learning_rate 6e-4 \
--warmup_ratio 0.1 \
--check_val_every_n_epoch 20 \
--lr_scheduler_type linear \
--max_length 820 \
--generation_mode \
--max_new_tokens 512 \
--dataset_name lm