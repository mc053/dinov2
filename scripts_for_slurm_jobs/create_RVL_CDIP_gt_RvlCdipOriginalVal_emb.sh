#!/bin/bash
#SBATCH --job-name create-rvl-cdip-gt-rvlcdiporiginalval-emb
#SBATCH --output create-rvl-cdip-gt-rvlcdiporiginalval-emb-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=tars
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Start creating RVL_CDIP_gt_RvlCdipOriginalVal_emb on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/dinov2_env/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python embeddings.py --model RVL_CDIP_gt --input RvlCdipOriginalVal
"