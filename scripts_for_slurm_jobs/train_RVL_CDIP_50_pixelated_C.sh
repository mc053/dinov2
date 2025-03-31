#!/bin/bash
#SBATCH --job-name train-rvl-cdip-50-pixelated-c
#SBATCH --output train-rvl-cdip-50-pixelated-c-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=tars
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting RVL_CDIP_50_pixelated_C training on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/dinov2_env/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python dinov2/train/train.py --config-file dinov2/configs/train/rvl_cdip_50_pixelated_c.yaml --output-dir RVL_CDIP_50_pixelated_C
"

# when starting with run/train/train.py --nodes 1 --gpus 1