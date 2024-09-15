#!/bin/sh

#SBATCH -J mollarge-final-t2m-reason
#SBATCH --exclude=node76,node54,node79
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

srun python model/reasoning_generator.py \
--architecture molt5-large \
--cot_mode multiset_formula-chain-aromatic-con_ring_name-func_simple \
--wandb_mode online \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 250 \
--model_id laituan245 \
--max_length 820 \
--generation_mode \
--max_new_tokens 256 \
--check_val_every_n_epoch 10 \
--run_id 5a6kvjyo





