#!/bin/bash
###############################################################################
# Run All Tests — Revised (Target-Disjoint Split)
#
# Generates all figures and evaluation outputs using the model trained
# on the target-disjoint data split.
#
# Usage:
#   ./test/run_all_tests_revs.sh --checkpoint <path_to_checkpoint>
#
# Example:
#   ./test/run_all_tests_revs.sh \
#       --checkpoint pe_uncert_models/logs/crispAIPE_revs_conf/2026-02-21-00-41-59/best_model-*.ckpt
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [ -f "/Users/furkanozden/aidev/bin/activate" ]; then
    source /Users/furkanozden/aidev/bin/activate
fi

PYTHON_CMD="${PYTHON_CMD:-python3}"
CONFIG="pe_uncert_models/configs/crispAIPE_revs_conf.json"
OUTPUT_DIR="test/figures_revs"
CHECKPOINT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: --checkpoint is required"
    echo "Usage: $0 --checkpoint <path_to_best_model.ckpt>"
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "Running All Tests — Revised (Target-Disjoint Split)"
echo "================================================================================"
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output Dir: $OUTPUT_DIR"
echo "================================================================================"
echo ""

run_test() {
    local name="$1"
    local script="$2"
    shift 2
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "Running: $name"
    echo "────────────────────────────────────────────────────────────────"
    $PYTHON_CMD "$script" "$@" && echo "  ✓ $name completed" || echo "  ✗ $name FAILED (continuing)"
    echo ""
}

# 1. Spearman correlations
run_test "Spearman Correlations" test/spearman_correlations.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" --calc_all

# 2. Diagnostics plot
run_test "Diagnostics Plot" test/diagnostics_plot.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR"

# 3. Diagnostics plot with baselines
run_test "Diagnostics Plot (with Baselines)" test/diagnostics_plot_with_baselines.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR"

# 4. Uncertainty analysis figure
run_test "Uncertainty Analysis Figure" test/uncertainty_analysis_figure.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR"

# 5. Outcome confidence analysis
run_test "Outcome Confidence Analysis" test/outcome_confidence_analysis.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR"

# 6. Dense distribution & confidence examples figure
run_test "Dense Distribution Examples" test/dense_dist_conf_examples_fig.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR"

# 7. Sample distributions
run_test "Sample Distributions" test/sample_distributions.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR"

# 8. Outcome prediction performance comparison
run_test "Outcome Prediction Performance Comparison" test/outcome_prediction_performance_comparison.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR/performance_comparison"

# 9. Attention interpretability
run_test "Attention Interpretability" test/attention_interpretability.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR"

# 10. Attention outcomes analysis
run_test "Attention Outcomes Analysis" test/attention_outcomes_analysis.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "================================================================================"
echo "All tests completed! Figures saved to: $OUTPUT_DIR"
echo "================================================================================"
