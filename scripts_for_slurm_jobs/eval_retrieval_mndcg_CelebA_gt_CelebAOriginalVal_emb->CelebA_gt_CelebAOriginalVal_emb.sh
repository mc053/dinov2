#!/bin/bash
#SBATCH --job-name eval-retrieval-mndcg-CelebA_gt_CelebAOriginalVal_emb->CelebA_gt_CelebAOriginalVal_emb
#SBATCH --output eval-retrieval-mndcg-CelebA_gt_CelebAOriginalVal_emb->CelebA_gt_CelebAOriginalVal_emb-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1

# Print some node information
echo "$(date)"
echo "Starting CelebA retrieval mnDCG evaluation for scenario "Ground Truth" on partition: GPU"
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
    --query CelebA_gt_CelebAOriginalVal_emb.json \
    --database CelebA_gt_CelebAOriginalVal_emb.json
"