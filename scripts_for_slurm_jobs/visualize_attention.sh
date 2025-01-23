#!/bin/bash
#SBATCH --job-name visualize-attention
#SBATCH --output visualize-attention-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1

# Print some node information
echo "$(date)"
echo "Visualize attention on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/pia11_clean/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python visualize_attention.py --model RVL_CDIP_gt
"

# when starting with run/train/train.py --nodes 1 --gpus 1