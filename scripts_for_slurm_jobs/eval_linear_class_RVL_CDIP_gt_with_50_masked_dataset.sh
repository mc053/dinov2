#!/bin/bash
#SBATCH --job-name eval-linear-class-rvl-cdip-gt-with-50-masked-dataset
#SBATCH --output eval-linear-class-rvl-cdip-gt-with-50-masked-dataset-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=tars
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting RVL_CDIP_gt linear class evaluation with 50% masked train and val dataset on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/dinov2_env/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python dinov2/eval/linear.py \
    --config-file RVL_CDIP_gt/config.yaml \
    --pretrained-weights RVL_CDIP_gt/eval/training_124999/teacher_checkpoint.pth \
    --output-dir RVL_CDIP_gt/eval/training_124999/linear_class_with_50_masked_dataset \
    --train-dataset RvlCdip50MaskedTrain \
    --val-dataset RvlCdip50MaskedVal
"