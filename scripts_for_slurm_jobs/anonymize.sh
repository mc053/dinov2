#!/bin/bash
#SBATCH --job-name anonymize-celeba-val-blurred
#SBATCH --output anonymize-celeba-val-blurred-%j.out
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting CelebA val set anonymization with gaussian blurring on partition: CPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
# echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w paddle bash -c "
    source /home/stud/m/mc085/mounted_home/pia11_clean/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    python -m dinov2.data.anonymizations
"

# tar -xzf /home/stud/m/mc085/mounted_home/pia11_clean.tar.gz -C /home/stud/m/mc085/mounted_home/pia11_clean &&