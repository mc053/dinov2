#!/bin/bash
#SBATCH --job-name eval-linear-gender-celeba-pixelated-c-with-pixelated-dataset
#SBATCH --output eval-linear-gender-celeba-pixelated-c-with-pixelated-dataset-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=tars
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting CelebA_pixelated_C linear gender evaluation with pixelated val dataset on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/dinov2_env/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python dinov2/eval/linear.py \
    --config-file CelebA_pixelated_C/config.yaml \
    --pretrained-weights CelebA_pixelated_C/eval/training_124999/teacher_checkpoint.pth \
    --output-dir CelebA_pixelated_C/eval/training_124999/linear_gender_with_pixelated_val_dataset \
    --train-dataset CelebAPixelatedTrain \
    --val-dataset CelebAPixelatedVal
"