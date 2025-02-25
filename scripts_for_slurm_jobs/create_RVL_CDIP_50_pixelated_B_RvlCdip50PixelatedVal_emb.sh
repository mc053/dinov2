#!/bin/bash
#SBATCH --job-name create-rvl-cdip-50-pixelated-b-rvlcdip50pixelatedval-emb
#SBATCH --output create-rvl-cdip-50-pixelated-b-rvlcdip50pixelatedval-emb-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Start creating RVL_CDIP_50_pixelated_B_RvlCdip50PixelatedVal_emb on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/dinov2_env/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python embeddings.py --model RVL_CDIP_50_pixelated_B --input RvlCdip50PixelatedVal
"