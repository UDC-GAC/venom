#!/bin/bash

#SBATCH --job-name=sparsifier
#SBATCH --mem=128GB

#SBATCH --time=4-00:00:00
#SBATCH --partition=amdrtx

module load cuda/11.7.1
source activate sparseml_artf

echo "SLURM_JOB_NUM_NODES $SLURM_JOB_NUM_NODES"
echo "HOSTNAME $HOSTNAME"

echo "Args train.sh: $@"

# vw_8
srun integrations/huggingface-transformers/scripts/30epochs_gradual_pruning_squad_block8_875.sh

# 128:2:16 (pair-wise)
srun integrations/huggingface-transformers/scripts/obs216v128_squad_gradual_pair.sh

# 1:2:16 (pair-wise)
srun integrations/huggingface-transformers/scripts/obs216_squad_gradual_pair.sh