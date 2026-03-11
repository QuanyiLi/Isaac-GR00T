#!/bin/bash
#SBATCH --job-name=gr00t_finetune
#SBATCH --output=/home/lfeng/task_logs/%j.log
#SBATCH --partition=h100
#SBATCH --qos=vita
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=200G

# Dataset path on remote machine
DATASET_PATH="/work/vita/lanfeng/vlas/wise_dataset_0.4.3/merged_train_v2.1"
OUTPUT_DIR="/work/vita/lanfeng/vlas/exp_results/gr00t_finetune_$(date +%Y%m%d_%H%M%S)"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501

# Launch the training using launch_finetune.py
# This script automatically registers the modality config and sets fine-tuning defaults.
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /work/vita/lanfeng/vlas/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path $DATASET_PATH \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /work/vita/lanfeng/vlas/Isaac-GR00T/modality_config.py \
    --num-gpus 4 \
    --output-dir $OUTPUT_DIR \
    --save-total-limit 5 \
    --save-steps 500 \
    --max-steps 10000 \
    --use-wandb \
    --global-batch-size 128 \
    --dataloader-num-workers 4
