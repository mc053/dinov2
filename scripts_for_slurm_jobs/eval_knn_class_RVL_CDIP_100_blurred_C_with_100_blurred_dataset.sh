#!/bin/bash
#SBATCH --job-name eval-knn-class-rvl-cdip-100-blurred-c-with-100-blurred-dataset
#SBATCH --output eval-knn-class-rvl-cdip-100-blurred-c-with-100-blurred-dataset-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=tars
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting RVL_CDIP_100_blurred_C knn class evaluation with 100% blurred train and val dataset on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/dinov2_env/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python dinov2/eval/knn.py \
    --config-file RVL_CDIP_100_blurred_C/config.yaml \
    --pretrained-weights RVL_CDIP_100_blurred_C/eval/training_124999/teacher_checkpoint.pth \
    --output-dir RVL_CDIP_100_blurred_C/eval/training_124999/knn_class_with_100_blurred_dataset \
    --train-dataset RvlCdip100BlurredTrain \
    --val-dataset RvlCdip100BlurredVal
"