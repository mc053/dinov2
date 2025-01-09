#!/bin/bash
#SBATCH --job-name anonymize-rvl-cdip-val-mask-out-25
#SBATCH --output anonymize-rvl-cdip-val-mask-out-25-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting RVL-CDIP val set anonymization with mask out (25%) on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w paddle bash -c "
    source /home/stud/m/mc085/mounted_home/pia11_clean/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    python -m dinov2.data.anonymizations
"

# tar -xzf /home/stud/m/mc085/mounted_home/pia11_clean.tar.gz -C /home/stud/m/mc085/mounted_home/pia11_clean &&