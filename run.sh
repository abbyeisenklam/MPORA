#!/bin/bash

TASKTYPE="spec"
DELTAS=(0.01 0.02 0.03 0.04 0.05)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEATURES_FILE="$SCRIPT_DIR/${TASKTYPE}_input_feature_map.txt"

FIRST_DELTA=true
for DELTA in "${DELTAS[@]}"; do
    echo "========================================"
    echo "DELTA = $DELTA"
    echo "========================================"
    while IFS=',' read -r taskname rest || [[ -n "$taskname" ]]; do
        [[ -z "$taskname" || "$taskname" == \#* ]] && continue
        input_features="${rest}"
        PLOT_RUNS=""
        # only plot training runs for the first delta to avoid regenerating the same plots for each delta
        [[ "$FIRST_DELTA" == true ]] && PLOT_RUNS="--plot-train-runs"
        python train.py --task "$taskname" --task-type "$TASKTYPE" --window-size-delta "$DELTA" --train-split 0.1 --plot-nmae-histogram $PLOT_RUNS --input-features "$input_features"
        echo "Done training and testing task: $taskname (delta=$DELTA)"
    done < "$FEATURES_FILE"
    FIRST_DELTA=false
done

echo "========================================"
echo "Plotting NMAE vs. delta"
echo "========================================"
python plot_delta_vs_metric.py --metric nmae

echo "========================================"
echo "Weighted conformal prediction (delta=0.01)"
echo "========================================"
python train_weight_net.py

exit 0
