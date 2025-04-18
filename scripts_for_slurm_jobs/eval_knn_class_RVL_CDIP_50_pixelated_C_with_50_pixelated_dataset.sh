#!/bin/bash
#SBATCH --job-name eval-knn-class-rvl-cdip-50-pixelated-c-with-50-pixelated-dataset
#SBATCH --output eval-knn-class-rvl-cdip-50-pixelated-c-with-50-pixelated-dataset-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=tars
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting RVL_CDIP_50_pixelated_C knn class evaluation with 50% pixelated train and val dataset on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/dinov2_env/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python dinov2/eval/knn.py \
    --config-file RVL_CDIP_50_pixelated_C/config.yaml \
    --pretrained-weights RVL_CDIP_50_pixelated_C/eval/training_124999/teacher_checkpoint.pth \
    --output-dir RVL_CDIP_50_pixelated_C/eval/training_124999/knn_class_with_50_pixelated_dataset \
    --train-dataset RvlCdip50PixelatedTrain \
    --val-dataset RvlCdip50PixelatedVal
"