#!/bin/sh

#SBATCH -J reason-large-ra
#SBATCH -p A100-80GB
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
--cot_mode_multiset None \
--cot_mode_ring \
--cot_mode_aromatic \
--wandb_mode online \
--train_batch_size 2 \
--eval_batch_size 2 \
--epochs 250 \
--max_length 512 \
--run_id mr0ekg5x



