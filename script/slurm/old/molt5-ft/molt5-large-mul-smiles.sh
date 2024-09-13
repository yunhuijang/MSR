#!/bin/sh

#SBATCH -J ft-large-mul-smiles
#SBATCH -q add_hpgpu
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
--cot_mode multiset_formula-func_smiles-chain-aromatic-con_ring_name \
--wandb_mode online \
--train_batch_size 4 \
--eval_batch_size 4 \
--check_val_every_n_epoch 5 \
--epochs 250 \
--model_id laituan245 \
--max_length 820 



