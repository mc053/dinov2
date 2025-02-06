#!/bin/bash
#SBATCH --job-name eval-retrieval-mndcg-5percent-CelebA_masked_C_CelebAOriginalVal_emb->CelebA_masked_C_CelebAMaskedVal_emb
#SBATCH --output eval-retrieval-mndcg-5percent-CelebA_masked_C_CelebAOriginalVal_emb->CelebA_masked_C_CelebAMaskedVal_emb-%j.out
#SBATCH --cpus-per-task 4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting CelebA retrieval mnDCG evaluation for scenario "Adaption C/Mask with Unanonymized Query Image" on partition: GPU"
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
    --query CelebA_masked_C_CelebAOriginalVal_emb.json \
    --database CelebA_masked_C_CelebAMaskedVal_emb.json
"