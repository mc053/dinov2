#!/bin/bash
#SBATCH --job-name eval-knn-rvl-cdip-gt-with-original-val-dataset
#SBATCH --output eval-knn-rvl-cdip-gt-with-original-val-dataset-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=tars
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting RVL_CDIP_gt knn evaluation with original val dataset on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/dinov2_env/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python dinov2/eval/knn.py \
    --config-file RVL_CDIP_gt/config.yaml \
    --pretrained-weights RVL_CDIP_gt/eval/training_124999/teacher_checkpoint.pth \
    --output-dir RVL_CDIP_gt/eval/training_124999/knn_with_original_val_dataset \
    --train-dataset RvlCdipOriginalTrain \
    --val-dataset RvlCdipOriginalVal
"