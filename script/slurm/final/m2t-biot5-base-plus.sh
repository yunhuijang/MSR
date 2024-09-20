#!/bin/sh

#SBATCH -J m2t-biot5-base-plus
#SBATCH --exclude=n76,n56,n54,n52
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

srun python model/one_stage_generator_mol2text.py \
--architecture biot5-base-plus \
--cot_mode multiset_formula-chain-aromatic-con_ring_name-func_simple-chiral \
--wandb_mode online \
--train_batch_size 16 \
--eval_batch_size 16 \
--epochs 250 \
--model_id QizhiPei \
--weight_decay 0 \
--learning_rate 1e-3 \
--warmup_ratio 0.1 \
--check_val_every_n_epoch 20 \
--lr_scheduler_type cosine \
--max_length 820 \
--generation_mode \
--max_new_tokens 512




