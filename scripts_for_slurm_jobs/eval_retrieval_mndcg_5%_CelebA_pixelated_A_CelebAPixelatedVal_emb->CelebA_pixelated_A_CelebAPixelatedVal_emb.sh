#!/bin/bash
#SBATCH --job-name eval-retrieval-mndcg-5percent-CelebA_pixelated_A_CelebAPixelatedVal_emb->CelebA_pixelated_A_CelebAPixelatedVal_emb
#SBATCH --output eval-retrieval-mndcg-5percent-CelebA_pixelated_A_CelebAPixelatedVal_emb->CelebA_pixelated_A_CelebAPixelatedVal_emb-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=hal9k
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting CelebA retrieval mnDCG evaluation for scenario "Adaption A/Pixel with Anonymized Query Image" on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/pia11_clean/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python mean_normalized_discounted_cumulative_gain.py \
    --gt CelebA_retrieval_ground_truths.pkl \
    --percent 5 \
    --query CelebA_pixelated_A_CelebAPixelatedVal_emb.json \
    --database CelebA_pixelated_A_CelebAPixelatedVal_emb.json
"