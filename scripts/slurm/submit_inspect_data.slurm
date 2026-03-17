#!/bin/bash
#SBATCH --job-name=gr00t_inspect_data
#SBATCH --output=/home/lfeng/task_logs/%j_inspect.log
#SBATCH --partition=h100
#SBATCH --qos=vita
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=0
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G

# ─── Paths (adjust as needed) ───
DATASET_PATH="/work/vita/lanfeng/vlas/wise_dataset_0.4.3/merged_train_v2.1"
MODALITY_CONFIG="/work/vita/lanfeng/vlas/Isaac-GR00T/modality_config.py"
SCRIPT_PATH="/work/vita/lanfeng/vlas/Isaac-GR00T/scripts/inspect_single_datapoint.py"
OUTPUT_DIR="/work/vita/lanfeng/vlas/data_inspection/inspect_$(date +%Y%m%d_%H%M%S)"

# ─── Which datapoint to inspect ───
EPISODE_INDEX=0
STEP_INDEX=5

export PYTHONPATH="/work/vita/lanfeng/vlas/vla:/work/vita/lanfeng/vlas/Isaac-GR00T:$PYTHONPATH"

echo "=== GR00T Data Inspection ==="
echo "Output dir: $OUTPUT_DIR"
echo "Episode: $EPISODE_INDEX, Step: $STEP_INDEX"
echo ""

python $SCRIPT_PATH \
    --dataset-path $DATASET_PATH \
    --modality-config-path $MODALITY_CONFIG \
    --output-dir $OUTPUT_DIR \
    --video-backend pyav \
    --episode-index $EPISODE_INDEX \
    --step-index $STEP_INDEX \
    --embodiment-tag NEW_EMBODIMENT

echo ""
echo "=== Done. Outputs in: $OUTPUT_DIR ==="
