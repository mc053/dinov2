#!/bin/bash
#SBATCH --job-name eval-knn-gender-celeba-gt-with-blurred-dataset
#SBATCH --output eval-knn-gender-celeba-gt-with-blurred-dataset-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=tars
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting CelebA_gt knn gender evaluation with blurred train and val dataset on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/dinov2_env/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python dinov2/eval/knn.py \
    --config-file CelebA_gt/config.yaml \
    --pretrained-weights CelebA_gt/eval/training_124999/teacher_checkpoint.pth \
    --output-dir CelebA_gt/eval/training_124999/knn_gender_with_blurred_dataset \
    --train-dataset CelebABlurredTrain \
    --val-dataset CelebABlurredVal
"