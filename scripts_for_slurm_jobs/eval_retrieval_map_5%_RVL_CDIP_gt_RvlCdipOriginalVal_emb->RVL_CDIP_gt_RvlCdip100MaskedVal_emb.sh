#!/bin/bash
#SBATCH --job-name eval-retrieval-map-5percent-RVL_CDIP_gt_RvlCdipOriginalVal_emb->RVL_CDIP_gt_RvlCdip100MaskedVal_emb
#SBATCH --output eval-retrieval-map-5percent-RVL_CDIP_gt_RvlCdipOriginalVal_emb->RVL_CDIP_gt_RvlCdip100MaskedVal_emb-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting RVL CDIP retrieval MAP evaluation for scenario "Unadapted/Masked" with Unanonymized Query Image" on partition: GPU"
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
    --query CelebA_pixelated_C_CelebAPixelatedVal_emb.json \
    --database CelebA_pixelated_C_CelebAPixelatedVal_emb.json
"