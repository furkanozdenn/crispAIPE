#!/bin/bash

###############################################################################
# crispAIPE Training Script with Train-Test Split
# 
# This script allows you to train the crispAIPE model with customizable
# hyperparameters and data paths. All parameters can be passed as command-line
# arguments.
#
# Usage:
#   ./train_crispAIPE.sh [OPTIONS]
#
# Example:
#   ./train_crispAIPE.sh \
#     --train_data_path data/pridict_data/pridict-train.csv \
#     --test_data_path data/pridict_data/pridict-test.csv \
#     --batch_size 64 \
#     --lr 0.001 \
#     --epochs 50
###############################################################################

set -e  # Exit on error

# Activate virtual environment if it exists
if [ -f "/Users/furkanozden/aidev/bin/activate" ]; then
    source /Users/furkanozden/aidev/bin/activate
    echo "Activated virtual environment: /Users/furkanozden/aidev"
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find Python executable
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3."
    exit 1
fi

# Default values for all parameters
# Model Parameters
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

# Data Parameters
DATA="pridict-90k-cleaned"
PROJECT_NAME="crispAIPE_train_test_split"
VOCAB_PATH="pe_uncert_models/vocab/vocab_3mer.txt"
VOCAB_CHAR_DICT="ACGTN"
TRAIN_DATA_PATH="data/pridict_data/pridict-train.csv"
TEST_DATA_PATH="data/pridict_data/pridict-test.csv"
PEGRNA_LENGTH=99
SEQUENCE_LENGTH=99
TARGET_SEQ_FLANK_LEN=0
TARGET_SEQ_LEN=99
VAL_SPLIT=0.1
BATCH_SIZE=128
EPOCHS=100

# Training Parameters
LR=6e-4
DECAY_LR=true
WEIGHT_DECAY=0.01
WARMUP_EPOCHS=10
BETA1=0.9
BETA2=0.98
GRAD_CLIP=1.0
MIN_LR=1e-5
VALID_SPLIT=0.2
EARLY_STOP_PATIENCE=8
LOG_DIR="pe_uncert_models/logs"
EARLY_STOPPING=true
PATIENCE=8
CPU=false
MAX_EPOCHS=100
GPUS=""

# Function to display help
show_help() {
    cat << EOF
crispAIPE Training Script

MODEL PARAMETERS:
  --target_dna_flank_len INT        Target DNA flank length (default: $TARGET_DNA_FLANK_LEN)
  --kmer_size INT                   K-mer size (default: $KMER_SIZE)
  --char_dict_len INT               Character dictionary length (default: $CHAR_DICT_LEN)
  --n_embd INT                      Embedding dimension (default: $N_EMBD)
  --d_hid INT                       Hidden dimension (default: $D_HID)
  --d_model INT                     Model dimension (default: $D_MODEL)
  --n_layer INT                     Number of layers (default: $N_LAYER)
  --dropout FLOAT                   Dropout rate (default: $DROPOUT)
  --d_out_hid INT                   Output hidden dimension (default: $D_OUT_HID)
  --input_dim INT                   Input dimension (default: $INPUT_DIM)
  --latent_dim INT                  Latent dimension (default: $LATENT_DIM)
  --hidden_dim INT                  Hidden dimension (default: $HIDDEN_DIM)
  --embedding_dim INT               Embedding dimension (default: $EMBEDDING_DIM)
  --layers INT                      Number of layers (default: $LAYERS)
  --nhead INT                       Number of attention heads (default: $NHEAD)
  --bottleneck_dim INT              Bottleneck dimension (default: $BOTTLENECK_DIM)
  --assesor_type STR                Assessor type: multinomial, softmax, logit_normal (default: $ASSESOR_TYPE)

DATA PARAMETERS:
  --data STR                        Dataset name (default: $DATA)
  --project_name STR                Project name for wandb (default: $PROJECT_NAME)
  --vocab_path STR                  Path to vocabulary file (default: $VOCAB_PATH)
  --vocab_char_dict STR             Character dictionary (default: $VOCAB_CHAR_DICT)
  --train_data_path STR             Path to training data CSV (default: $TRAIN_DATA_PATH)
  --test_data_path STR              Path to test data CSV (default: $TEST_DATA_PATH)
  --pegrna_length INT               PEGRNA length (default: $PEGRNA_LENGTH)
  --sequence_length INT             Sequence length (default: $SEQUENCE_LENGTH)
  --target_seq_flank_len INT        Target sequence flank length (default: $TARGET_SEQ_FLANK_LEN)
  --target_seq_len INT              Target sequence length (default: $TARGET_SEQ_LEN)
  --val_split FLOAT                 Validation split ratio (default: $VAL_SPLIT)
  --batch_size INT                  Batch size (default: $BATCH_SIZE)
  --epochs INT                      Number of epochs (default: $EPOCHS)

TRAINING PARAMETERS:
  --lr FLOAT                        Learning rate (default: $LR)
  --decay_lr BOOL                   Enable learning rate decay (default: $DECAY_LR)
  --weight_decay FLOAT              Weight decay (default: $WEIGHT_DECAY)
  --warmup_epochs INT               Warmup epochs (default: $WARMUP_EPOCHS)
  --beta1 FLOAT                     Adam beta1 (default: $BETA1)
  --beta2 FLOAT                     Adam beta2 (default: $BETA2)
  --grad_clip FLOAT                 Gradient clipping value (default: $GRAD_CLIP)
  --min_lr FLOAT                    Minimum learning rate (default: $MIN_LR)
  --valid_split FLOAT               Validation split (default: $VALID_SPLIT)
  --early_stop_patience INT         Early stopping patience (default: $EARLY_STOP_PATIENCE)
  --log_dir STR                     Log directory (default: $LOG_DIR)
  --early_stopping BOOL             Enable early stopping (default: $EARLY_STOPPING)
  --patience INT                    Patience for early stopping (default: $PATIENCE)
  --cpu BOOL                        Use CPU only (default: $CPU)
  --max_epochs INT                  Maximum epochs (default: $MAX_EPOCHS)
  --gpus STR                        GPU IDs (comma-separated, e.g., "0,1") (default: auto-detect)

GENERAL:
  -h, --help                        Show this help message

Examples:
  # Basic training with default parameters
  ./train_crispAIPE.sh

  # Custom learning rate and batch size
  ./train_crispAIPE.sh --lr 0.001 --batch_size 64

  # Custom data paths
  ./train_crispAIPE.sh \\
    --train_data_path data/my_train.csv \\
    --test_data_path data/my_test.csv

  # Full custom configuration
  ./train_crispAIPE.sh \\
    --train_data_path data/pridict_data/pridict-train.csv \\
    --test_data_path data/pridict_data/pridict-test.csv \\
    --batch_size 64 \\
    --lr 0.001 \\
    --epochs 50 \\
    --dropout 0.2 \\
    --n_layer 6 \\
    --embedding_dim 128 \\
    --early_stopping true \\
    --patience 10

EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        # Model parameters
        --target_dna_flank_len)
            TARGET_DNA_FLANK_LEN="$2"
            shift 2
            ;;
        --kmer_size)
            KMER_SIZE="$2"
            shift 2
            ;;
        --char_dict_len)
            CHAR_DICT_LEN="$2"
            shift 2
            ;;
        --n_embd)
            N_EMBD="$2"
            shift 2
            ;;
        --d_hid)
            D_HID="$2"
            shift 2
            ;;
        --d_model)
            D_MODEL="$2"
            shift 2
            ;;
        --n_layer)
            N_LAYER="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --d_out_hid)
            D_OUT_HID="$2"
            shift 2
            ;;
        --input_dim)
            INPUT_DIM="$2"
            shift 2
            ;;
        --latent_dim)
            LATENT_DIM="$2"
            shift 2
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --embedding_dim)
            EMBEDDING_DIM="$2"
            shift 2
            ;;
        --layers)
            LAYERS="$2"
            shift 2
            ;;
        --nhead)
            NHEAD="$2"
            shift 2
            ;;
        --bottleneck_dim)
            BOTTLENECK_DIM="$2"
            shift 2
            ;;
        --assesor_type)
            ASSESOR_TYPE="$2"
            shift 2
            ;;
        # Data parameters
        --data)
            DATA="$2"
            shift 2
            ;;
        --project_name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --vocab_path)
            VOCAB_PATH="$2"
            shift 2
            ;;
        --vocab_char_dict)
            VOCAB_CHAR_DICT="$2"
            shift 2
            ;;
        --train_data_path)
            TRAIN_DATA_PATH="$2"
            shift 2
            ;;
        --test_data_path)
            TEST_DATA_PATH="$2"
            shift 2
            ;;
        --pegrna_length)
            PEGRNA_LENGTH="$2"
            shift 2
            ;;
        --sequence_length)
            SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        --target_seq_flank_len)
            TARGET_SEQ_FLANK_LEN="$2"
            shift 2
            ;;
        --target_seq_len)
            TARGET_SEQ_LEN="$2"
            shift 2
            ;;
        --val_split)
            VAL_SPLIT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            MAX_EPOCHS="$2"
            shift 2
            ;;
        # Training parameters
        --lr)
            LR="$2"
            shift 2
            ;;
        --decay_lr)
            DECAY_LR="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --warmup_epochs)
            WARMUP_EPOCHS="$2"
            shift 2
            ;;
        --beta1)
            BETA1="$2"
            shift 2
            ;;
        --beta2)
            BETA2="$2"
            shift 2
            ;;
        --grad_clip)
            GRAD_CLIP="$2"
            shift 2
            ;;
        --min_lr)
            MIN_LR="$2"
            shift 2
            ;;
        --valid_split)
            VALID_SPLIT="$2"
            shift 2
            ;;
        --early_stop_patience)
            EARLY_STOP_PATIENCE="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --early_stopping)
            EARLY_STOPPING="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --cpu)
            CPU="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required files exist
if [[ ! -f "$TRAIN_DATA_PATH" ]]; then
    echo "Error: Training data file not found: $TRAIN_DATA_PATH"
    exit 1
fi

if [[ ! -f "$TEST_DATA_PATH" ]]; then
    echo "Error: Test data file not found: $TEST_DATA_PATH"
    exit 1
fi

if [[ ! -f "$VOCAB_PATH" ]]; then
    echo "Warning: Vocabulary file not found: $VOCAB_PATH"
    echo "Continuing anyway..."
fi

# Create temporary config file
TEMP_CONFIG=$(mktemp /tmp/crispAIPE_config_XXXXXX.json)
trap "rm -f $TEMP_CONFIG" EXIT  # Clean up on exit

# Convert boolean strings to JSON booleans
if [[ "$DECAY_LR" == "true" ]] || [[ "$DECAY_LR" == "True" ]] || [[ "$DECAY_LR" == "1" ]]; then
    DECAY_LR_JSON="true"
else
    DECAY_LR_JSON="false"
fi

if [[ "$EARLY_STOPPING" == "true" ]] || [[ "$EARLY_STOPPING" == "True" ]] || [[ "$EARLY_STOPPING" == "1" ]]; then
    EARLY_STOPPING_JSON="true"
else
    EARLY_STOPPING_JSON="false"
fi

if [[ "$CPU" == "true" ]] || [[ "$CPU" == "True" ]] || [[ "$CPU" == "1" ]]; then
    CPU_JSON="true"
else
    CPU_JSON="false"
fi

# Generate JSON config file
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

# Print configuration summary
echo "================================================================================"
echo "crispAIPE Training Configuration"
echo "================================================================================"
echo "Training Data:    $TRAIN_DATA_PATH"
echo "Test Data:        $TEST_DATA_PATH"
echo "Batch Size:       $BATCH_SIZE"
echo "Learning Rate:    $LR"
echo "Epochs:           $MAX_EPOCHS"
echo "Dropout:          $DROPOUT"
echo "Early Stopping:   $EARLY_STOPPING (patience: $PATIENCE)"
echo "Project Name:     $PROJECT_NAME"
echo "Config File:      $TEMP_CONFIG"
echo "================================================================================"
echo ""

# Run training
echo "Starting training..."
echo ""

$PYTHON_CMD pe_uncert_models/models/train.py --config "$TEMP_CONFIG"

echo ""
echo "Training completed!"
echo "Config file was: $TEMP_CONFIG"
echo "Check logs in: $LOG_DIR/$PROJECT_NAME/"

