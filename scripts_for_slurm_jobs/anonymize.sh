#!/bin/bash
#SBATCH --job-name anonymize-rvl-cdip-train-blurring-50
#SBATCH --output anonymize-rvl-cdip-train-blurring-50-%j.out
#SBATCH --cpus-per-task 4

# Print some node information
echo "$(date)"
echo "Starting RVL-CDIP train set anonymization with blurring (50%) on partition: CPU"
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