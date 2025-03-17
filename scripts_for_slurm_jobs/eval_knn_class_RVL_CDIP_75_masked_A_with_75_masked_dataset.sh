#!/bin/bash
#SBATCH --job-name eval-knn-class-rvl-cdip-75-masked-a-with-75-masked-dataset
#SBATCH --output eval-knn-class-rvl-cdip-75-masked-a-with-75-masked-dataset-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting RVL_CDIP_75_masked_A knn class evaluation with 75% masked train and val dataset on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/dinov2_env/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python dinov2/eval/knn.py \
    --config-file RVL_CDIP_75_masked_A/config.yaml \
    --pretrained-weights RVL_CDIP_75_masked_A/eval/training_124999/teacher_checkpoint.pth \
    --output-dir RVL_CDIP_75_masked_A/eval/training_124999/knn_class_with_75_masked_dataset \
    --train-dataset RvlCdip75MaskedTrain \
    --val-dataset RvlCdip75MaskedVal
"