#!/bin/bash

###############################################################################
# crispAIPE Training Script — Revised (Target-Disjoint Split)
#
# Uses the target-disjoint data split from pridict_data_revs/ that keeps all
# pegRNAs targeting the same mutation in the same partition to prevent data
# leakage. Train/Val/Test are 70/15/15 split at the target-group level.
#
# Usage:
#   ./train_crispAIPE_revs.sh [OPTIONS]
#
# Example:
#   ./train_crispAIPE_revs.sh \
#     --train_data_path data/pridict_data_revs/pridict-train_revs.csv \
#     --val_data_path data/pridict_data_revs/pridict-val_revs.csv \
#     --test_data_path data/pridict_data_revs/pridict-test_revs.csv \
#     --batch_size 64 \
#     --lr 0.001 \
#     --epochs 50
###############################################################################

set -e

if [ -f "/Users/furkanozden/aidev/bin/activate" ]; then
    source /Users/furkanozden/aidev/bin/activate
    echo "Activated virtual environment: /Users/furkanozden/aidev"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3."
    exit 1
fi

# Default values
TARGET_DNA_FLANK_LEN=0
KMER_SIZE=3
CHAR_DICT_LEN=5
N_EMBD=64
D_HID=32
D_MODEL=16
N_LAYER=4
DROPOUT=0.1
D_OUT_HID=32
INPUT_DIM=5
LATENT_DIM=128
HIDDEN_DIM=64
EMBEDDING_DIM=64
LAYERS=4
NHEAD=4
BOTTLENECK_DIM=8
ASSESOR_TYPE="multinomial"

DATA="pridict-90k-cleaned"
PROJECT_NAME="crispAIPE_revs"
VOCAB_PATH="pe_uncert_models/vocab/vocab_3mer.txt"
VOCAB_CHAR_DICT="ACGTN"
TRAIN_DATA_PATH="data/pridict_data_revs/pridict-train_revs.csv"
VAL_DATA_PATH="data/pridict_data_revs/pridict-val_revs.csv"
TEST_DATA_PATH="data/pridict_data_revs/pridict-test_revs.csv"
PEGRNA_LENGTH=99
SEQUENCE_LENGTH=99
TARGET_SEQ_FLANK_LEN=0
TARGET_SEQ_LEN=99
VAL_SPLIT=0.15
BATCH_SIZE=128
EPOCHS=100

LR=6e-4
DECAY_LR=true
WEIGHT_DECAY=0.01
WARMUP_EPOCHS=10
BETA1=0.9
BETA2=0.98
GRAD_CLIP=1.0
MIN_LR=1e-5
VALID_SPLIT=0.15
EARLY_STOP_PATIENCE=8
LOG_DIR="pe_uncert_models/logs"
EARLY_STOPPING=true
PATIENCE=8
CPU=false
MAX_EPOCHS=100
GPUS=""

show_help() {
    cat << EOF
crispAIPE Revised Training Script (Target-Disjoint Split)

DATA PARAMETERS:
  --train_data_path STR     Path to training CSV (default: $TRAIN_DATA_PATH)
  --val_data_path STR       Path to validation CSV (default: $VAL_DATA_PATH)
  --test_data_path STR      Path to test CSV (default: $TEST_DATA_PATH)
  --batch_size INT          Batch size (default: $BATCH_SIZE)
  --epochs INT              Number of epochs (default: $EPOCHS)

TRAINING PARAMETERS:
  --lr FLOAT                Learning rate (default: $LR)
  --dropout FLOAT           Dropout rate (default: $DROPOUT)
  --early_stopping BOOL     Enable early stopping (default: $EARLY_STOPPING)
  --patience INT            Patience for early stopping (default: $PATIENCE)
  --cpu BOOL                Use CPU only (default: $CPU)

GENERAL:
  -h, --help                Show this help message

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --target_dna_flank_len) TARGET_DNA_FLANK_LEN="$2"; shift 2 ;;
        --kmer_size) KMER_SIZE="$2"; shift 2 ;;
        --char_dict_len) CHAR_DICT_LEN="$2"; shift 2 ;;
        --n_embd) N_EMBD="$2"; shift 2 ;;
        --d_hid) D_HID="$2"; shift 2 ;;
        --d_model) D_MODEL="$2"; shift 2 ;;
        --n_layer) N_LAYER="$2"; shift 2 ;;
        --dropout) DROPOUT="$2"; shift 2 ;;
        --d_out_hid) D_OUT_HID="$2"; shift 2 ;;
        --input_dim) INPUT_DIM="$2"; shift 2 ;;
        --latent_dim) LATENT_DIM="$2"; shift 2 ;;
        --hidden_dim) HIDDEN_DIM="$2"; shift 2 ;;
        --embedding_dim) EMBEDDING_DIM="$2"; shift 2 ;;
        --layers) LAYERS="$2"; shift 2 ;;
        --nhead) NHEAD="$2"; shift 2 ;;
        --bottleneck_dim) BOTTLENECK_DIM="$2"; shift 2 ;;
        --assesor_type) ASSESOR_TYPE="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        --project_name) PROJECT_NAME="$2"; shift 2 ;;
        --vocab_path) VOCAB_PATH="$2"; shift 2 ;;
        --vocab_char_dict) VOCAB_CHAR_DICT="$2"; shift 2 ;;
        --train_data_path) TRAIN_DATA_PATH="$2"; shift 2 ;;
        --val_data_path) VAL_DATA_PATH="$2"; shift 2 ;;
        --test_data_path) TEST_DATA_PATH="$2"; shift 2 ;;
        --pegrna_length) PEGRNA_LENGTH="$2"; shift 2 ;;
        --sequence_length) SEQUENCE_LENGTH="$2"; shift 2 ;;
        --target_seq_flank_len) TARGET_SEQ_FLANK_LEN="$2"; shift 2 ;;
        --target_seq_len) TARGET_SEQ_LEN="$2"; shift 2 ;;
        --val_split) VAL_SPLIT="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; MAX_EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --decay_lr) DECAY_LR="$2"; shift 2 ;;
        --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --warmup_epochs) WARMUP_EPOCHS="$2"; shift 2 ;;
        --beta1) BETA1="$2"; shift 2 ;;
        --beta2) BETA2="$2"; shift 2 ;;
        --grad_clip) GRAD_CLIP="$2"; shift 2 ;;
        --min_lr) MIN_LR="$2"; shift 2 ;;
        --valid_split) VALID_SPLIT="$2"; shift 2 ;;
        --early_stop_patience) EARLY_STOP_PATIENCE="$2"; shift 2 ;;
        --log_dir) LOG_DIR="$2"; shift 2 ;;
        --early_stopping) EARLY_STOPPING="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --cpu) CPU="$2"; shift 2 ;;
        --max_epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; echo "Use --help for usage information"; exit 1 ;;
    esac
done

for f in "$TRAIN_DATA_PATH" "$VAL_DATA_PATH" "$TEST_DATA_PATH"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: Data file not found: $f"
        exit 1
    fi
done

TEMP_CONFIG=$(mktemp "$SCRIPT_DIR/crispAIPE_revs_config_XXXXXX.json")
trap "rm -f $TEMP_CONFIG" EXIT

[[ "$DECAY_LR" =~ ^(true|True|1)$ ]] && DECAY_LR_JSON="true" || DECAY_LR_JSON="false"
[[ "$EARLY_STOPPING" =~ ^(true|True|1)$ ]] && EARLY_STOPPING_JSON="true" || EARLY_STOPPING_JSON="false"
[[ "$CPU" =~ ^(true|True|1)$ ]] && CPU_JSON="true" || CPU_JSON="false"

cat > "$TEMP_CONFIG" << EOF
{
    "model_parameters": {
        "target_dna_flank_len": $TARGET_DNA_FLANK_LEN,
        "kmer_size": $KMER_SIZE,
        "char_dict_len": $CHAR_DICT_LEN,
        "n_embd": $N_EMBD,
        "d_hid": $D_HID,
        "d_model": $D_MODEL,
        "n_layer": $N_LAYER,
        "dropout": $DROPOUT,
        "d_out_hid": $D_OUT_HID,
        "input_dim": $INPUT_DIM,
        "latent_dim": $LATENT_DIM,
        "hidden_dim": $HIDDEN_DIM,
        "embedding_dim": $EMBEDDING_DIM,
        "layers": $LAYERS,
        "nhead": $NHEAD,
        "bottleneck_dim": $BOTTLENECK_DIM,
        "assesor_type": "$ASSESOR_TYPE"
    },
    "data_parameters": {
        "data": "$DATA",
        "project_name": "$PROJECT_NAME",
        "vocab_path": "$VOCAB_PATH",
        "vocab_char_dict": "$VOCAB_CHAR_DICT",
        "train_data_path": "$TRAIN_DATA_PATH",
        "val_data_path": "$VAL_DATA_PATH",
        "test_data_path": "$TEST_DATA_PATH",
        "pegrna_length": $PEGRNA_LENGTH,
        "sequence_length": $SEQUENCE_LENGTH,
        "target_seq_flank_len": $TARGET_SEQ_FLANK_LEN,
        "target_seq_len": $TARGET_SEQ_LEN,
        "val_split": $VAL_SPLIT,
        "batch_size": $BATCH_SIZE,
        "epochs": $EPOCHS
    },
    "training_parameters": {
        "lr": $LR,
        "decay_lr": $DECAY_LR_JSON,
        "weight_decay": $WEIGHT_DECAY,
        "warmup_epochs": $WARMUP_EPOCHS,
        "beta1": $BETA1,
        "beta2": $BETA2,
        "grad_clip": $GRAD_CLIP,
        "min_lr": $MIN_LR,
        "valid_split": $VALID_SPLIT,
        "early_stop_patience": $EARLY_STOP_PATIENCE,
        "log_dir": "$LOG_DIR",
        "early_stopping": $EARLY_STOPPING_JSON,
        "patience": $PATIENCE,
        "cpu": $CPU_JSON,
        "max_epochs": $MAX_EPOCHS,
        "gpus": "$GPUS"
    }
}
EOF

echo "================================================================================"
echo "crispAIPE REVISED Training (Target-Disjoint Split)"
echo "================================================================================"
echo "Training Data:    $TRAIN_DATA_PATH"
echo "Validation Data:  $VAL_DATA_PATH"
echo "Test Data:        $TEST_DATA_PATH"
echo "Split Strategy:   Target-disjoint (70/15/15)"
echo "Batch Size:       $BATCH_SIZE"
echo "Learning Rate:    $LR"
echo "Epochs:           $MAX_EPOCHS"
echo "Dropout:          $DROPOUT"
echo "Early Stopping:   $EARLY_STOPPING (patience: $PATIENCE)"
echo "Project Name:     $PROJECT_NAME"
echo "================================================================================"
echo ""

echo "Starting training..."
echo ""

$PYTHON_CMD pe_uncert_models/models/train.py --config "$TEMP_CONFIG"

echo ""
echo "Training completed!"
echo "Check logs in: $LOG_DIR/$PROJECT_NAME/"
