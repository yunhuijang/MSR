#!/bin/sh

#SBATCH -J analysis
#SBATCH -q add_hpgpu
#SBATCH -p A100-80GB
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

srun python model/analyze_structure_failure.py \
--wandb_mode online \
--cot_mode_aromatic \
--cot_mode_chain \
--cot_mode_con_ring_name \
--cot_mode_functional_group




