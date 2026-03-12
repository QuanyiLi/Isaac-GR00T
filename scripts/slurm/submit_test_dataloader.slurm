#!/bin/bash
#SBATCH --job-name=test_dataloader
#SBATCH --output=/home/lfeng/task_logs/%j.log
#SBATCH --partition=h100
#SBATCH --qos=vita
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=64G

# Dataset path on remote machine
DATASET_PATH="/work/vita/lanfeng/vlas/wise_dataset_0.4.3/merged_train_v2.1"
OUTPUT_DIR="/work/vita/lanfeng/vlas/exp_results/test_dataloader_$(date +%Y%m%d_%H%M%S)"

# Run test script (single GPU, no torchrun needed)
python /work/vita/lanfeng/vlas/Isaac-GR00T/scripts/test_dataloader.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path $DATASET_PATH \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /work/vita/lanfeng/vlas/Isaac-GR00T/modality_config.py \
    --video-backend pyav \
    --num-gpus 1 \
    --output-dir $OUTPUT_DIR \
    --global-batch-size 2 \
    --dataloader-num-workers 0
