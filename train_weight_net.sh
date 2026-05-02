#!/bin/bash
# Train weighted conformal prediction weight networks for all benchmarks
# and evaluate per-scenario coverage. Results saved to results/wcp_results.txt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python train_weight_net.py \
    --n-clusters 4 \
    --delta      0.1 \
    --epochs     500 \
    --batch-size 512 \
    --lr         0.01 \
    --lambda_    1.0 \
    --beta       0.01 \
    --hidden-dims 64 32 8 \
    --device     auto \
    --verbose

echo ""
echo "Done. Results in results/wcp_results.txt"
