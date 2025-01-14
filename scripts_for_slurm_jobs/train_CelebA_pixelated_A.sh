#!/bin/bash
#SBATCH --job-name train-celeba-pixelated-a
#SBATCH --output train-celeba-pixelated-a-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=tars

# Print some node information
echo "$(date)"
echo "Starting CelebA_pixelated_A training on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/dinov2_env/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python dinov2/train/train.py --config-file dinov2/configs/train/celeba_pixelated_a.yaml --output-dir CelebA_pixelated_A
"