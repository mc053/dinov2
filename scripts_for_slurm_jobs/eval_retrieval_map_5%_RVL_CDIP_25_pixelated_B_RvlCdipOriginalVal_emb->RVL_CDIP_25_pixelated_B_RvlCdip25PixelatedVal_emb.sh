#!/bin/bash
#SBATCH --job-name eval-retrieval-map-5percent-RVL_CDIP_25_pixelated_B_RvlCdipOriginalVal_emb->RVL_CDIP_25_pixelated_B_RvlCdip25PixelatedVal_emb
#SBATCH --output eval-retrieval-map-5percent-RVL_CDIP_25_pixelated_B_RvlCdipOriginalVal_emb->RVL_CDIP_25_pixelated_B_RvlCdip25PixelatedVal_emb-%j.out
#SBATCH --cpus-per-task 4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting RVL CDIP retrieval mAP evaluation for scenario "Adaption B/Pixelated" (25%) with Unanonymized Query Image on partition: CPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

# Start jupyter lab
srun --unbuffered enroot start --mount $HOME:$HOME/mounted_home -w mc085 bash -c "
    source /home/stud/m/mc085/mounted_home/pia11_clean/bin/activate &&
    cd /home/stud/m/mc085/mounted_home/dinov2 &&
    PYTHONPATH=. python mean_average_precision.py \
    --gt RVL_CDIP \
    --percent 5 \
    --query RVL_CDIP_25_pixelated_B_RvlCdipOriginalVal_emb.json \
    --database RVL_CDIP_25_pixelated_B_RvlCdip25PixelatedVal_emb.json
"