#!/bin/bash
#SBATCH --job-name eval-retrieval-mndcg-5percent-RVL_CDIP_100_blurred_C_RvlCdipOriginalVal_emb->RVL_CDIP_100_blurred_C_RvlCdip100BlurredVal_emb
#SBATCH --output eval-retrieval-mndcg-5percent-RVL_CDIP_100_blurred_C_RvlCdipOriginalVal_emb->RVL_CDIP_100_blurred_C_RvlCdip100BlurredVal_emb-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=ada
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting RVL CDIP retrieval mnDCG evaluation for scenario "Adaption C/Blurred" (100%) with Unanonymized Query Image on partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/pia11_clean/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python mean_normalized_discounted_cumulative_gain.py \
    --gt RVL_CDIP \
    --percent 5 \
    --query RVL_CDIP_100_blurred_C_RvlCdipOriginalVal_emb.json \
    --database RVL_CDIP_100_blurred_C_RvlCdip100BlurredVal_emb.json
"