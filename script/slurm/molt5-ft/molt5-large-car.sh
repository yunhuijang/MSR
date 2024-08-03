#!/bin/sh

#SBATCH -J ft-large-car
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

srun python model/one_stage_generator.py \
--architecture molt5-large \
--cot_mode_multiset None \
--cot_mode_aromatic \
--cot_mode_ring \
--cot_mode_chain \
--wandb_mode online \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 250



