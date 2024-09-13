#!/bin/sh

#SBATCH -J reason-biobase-mul-form
#SBATCH -p A6000
#SBATCH --gres=gpu:4
#SBATCH -o sbatch_log/%x.out
#SBATCH -n 4

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
--architecture biot5-plus-base \
--cot_mode multiset_formula-func_smiles-chain-aromatic-con_ring_name \
--wandb_mode online \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 250 \
--model_id QizhiPei




