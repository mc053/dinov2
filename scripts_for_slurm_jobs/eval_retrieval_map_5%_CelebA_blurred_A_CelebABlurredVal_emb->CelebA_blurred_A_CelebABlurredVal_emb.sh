#!/bin/bash
#SBATCH --job-name eval-retrieval-map-5percent-CelebA_blurred_A_CelebABlurredVal_emb->CelebA_blurred_A_CelebABlurredVal_emb
#SBATCH --output eval-retrieval-map-5percent-CelebA_blurred_A_CelebABlurredVal_emb->CelebA_blurred_A_CelebABlurredVal_emb-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=tars
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting CelebA retrieval mAP evaluation for scenario "Adaption A/Blur with Anonymized Query Image" on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/pia11_clean/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python mean_average_precision.py \
    --gt CelebA_retrieval_ground_truths.pkl \
    --percent 5 \
    --query CelebA_blurred_A_CelebABlurredVal_emb.json \
    --database CelebA_blurred_A_CelebABlurredVal_emb.json
"