#!/bin/bash
#SBATCH --job-name a-training-test
#SBATCH --output a-training-test-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=tars

# Print some node information
echo "$(date)"
echo "Starting A Training Test on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/dinov2_env/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python dinov2/train/train.py --config-file dinov2/configs/train/celeba_pixelated_a.yaml --output-dir CelebA_pixelated_A
"

# when starting with run/train/train.py --nodes 1 --gpus 1