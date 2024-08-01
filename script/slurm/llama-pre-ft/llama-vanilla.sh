#!/bin/sh

#SBATCH -J ftpre-llama
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

srun python model/one_stage_generator_llama.py \
--architecture llama \
--cot_mode_multiset None \
--wandb_mode online \
--train_batch_size 4 \
--eval_batch_size 4 \
--gen_batch_size 32 \
--epochs 20 \
--max_length 512 \
--pretrain_model_id 1azi0wgu \
--check_val_every_n_epoch 1 \
--run_id cx59vec3


