#!/bin/sh

#SBATCH -J iupac_train_9000
#SBATCH --exclude=n76,n56,n54,n52
#SBATCH -p cpu-max16
#SBATCH -q nogpu
#SBATCH -n 1
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

srun python generate_iupac.py \
--split train \
--start_index 9000