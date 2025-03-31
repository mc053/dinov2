#!/bin/bash
#SBATCH --job-name eval-retrieval-map-5percent-RVL_CDIP_75_pixelated_C_RvlCdip75PixelatedVal_emb->RVL_CDIP_75_pixelated_C_RvlCdip75PixelatedVal_emb
#SBATCH --output eval-retrieval-map-5percent-RVL_CDIP_75_pixelated_C_RvlCdip75PixelatedVal_emb->RVL_CDIP_75_pixelated_C_RvlCdip75PixelatedVal_emb-%j.out
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodelist=hal9k
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mc085@hdm-stuttgart.de

# Print some node information
echo "$(date)"
echo "Starting RVL CDIP retrieval mAP evaluation for scenario "Adaption C/Pixelated" (75%) with Anonymized Query Image on partition: GPU"
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
    --query RVL_CDIP_75_pixelated_C_RvlCdip75PixelatedVal_emb.json \
    --database RVL_CDIP_75_pixelated_C_RvlCdip75PixelatedVal_emb.json
"