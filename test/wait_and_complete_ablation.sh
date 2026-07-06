#!/bin/bash
# Script to wait for training to complete and then run full ablation study

echo "============================================================"
echo "Waiting for ablation model training to complete..."
echo "============================================================"
echo ""

PROJECT_ROOT="/Users/furkanozden/Desktop/pe-uncert"
cd "$PROJECT_ROOT"

source /Users/furkanozden/aidev/bin/activate

# Check every 2 minutes for training completion
MAX_WAIT=7200  # 2 hours max wait
CHECK_INTERVAL=120  # 2 minutes
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    echo "Checking training status... (waited ${ELAPSED}s)"
    
    # Run the completion checker
    python test/complete_ablation_study.py
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "============================================================"
        echo "SUCCESS! Ablation study is complete!"
        echo "============================================================"
        exit 0
    fi
    
    echo ""
    echo "Training still in progress. Waiting ${CHECK_INTERVAL}s before next check..."
    sleep $CHECK_INTERVAL
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))
done

echo ""
echo "============================================================"
echo "Timeout reached. Training may still be in progress."
echo "Run 'python test/complete_ablation_study.py' manually"
echo "once training completes."
echo "============================================================"
exit 1


