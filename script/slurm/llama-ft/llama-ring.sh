#!/bin/sh

#SBATCH -J ft-llama-ring
#SBATCH -p A5000
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

srun python model/one_stage_generator_llama.py \
--architecture llama \
--cot_mode_multiset None \
--cot_mode_ring \
--wandb_mode online \
--train_batch_size 256 \
--eval_batch_size 4 \
--gen_batch_size 64 \
--epochs 250 \
--max_length 1024 \
--run_id 0q8vv7pn



