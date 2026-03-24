#!/bin/bash
set -e
# Usage: ./upload_results.sh <LOCAL_PORT> <LOCAL_IP>
# Example: ./upload_results.sh 13585 91.199.227.82
#
# Run this ON THE POD after training completes.
# It packages results and SCPs back to your local machine.

LOCAL_PORT="${1:-13585}"
LOCAL_IP="${2:-91.199.227.82}"
DEST_USER="root"
DEST_PATH="/workspace/results_$(date +%Y%m%d_%H%M%S)"

cd /workspace/parameter-golf

echo "=== Collecting submission files ==="

SUBMISSION_DIR="/workspace/submission_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SUBMISSION_DIR"

# Copy model files
cp final_model.int6.ptz "$SUBMISSION_DIR/" 2>/dev/null && echo "Copied final_model.int6.ptz" || echo "WARNING: final_model.int6.ptz not found"
cp final_model.pt "$SUBMISSION_DIR/" 2>/dev/null && echo "Copied final_model.pt" || echo "WARNING: final_model.pt not found (optional)"

# Copy training script
cp train_avery.py "$SUBMISSION_DIR/"

# Copy all seed logs
for seed in 1337 42 123; do
    if [ -f "train_seed${seed}.log" ]; then
        cp "train_seed${seed}.log" "$SUBMISSION_DIR/"
        echo "Copied train_seed${seed}.log"
    else
        echo "WARNING: train_seed${seed}.log not found"
    fi
done

# Copy logs directory if present
if [ -d "logs" ]; then
    cp -r logs "$SUBMISSION_DIR/"
    echo "Copied logs/"
fi

# Write score summary
echo "=== val_bpb scores ===" > "$SUBMISSION_DIR/scores.txt"
grep "final_int6.*val_bpb" train_seed*.log 2>/dev/null >> "$SUBMISSION_DIR/scores.txt" || true
grep "val_bpb" train_seed*.log 2>/dev/null | tail -3 >> "$SUBMISSION_DIR/scores.txt" || true
cat "$SUBMISSION_DIR/scores.txt"

echo ""
echo "=== Submission dir contents ==="
ls -lh "$SUBMISSION_DIR/"

echo ""
echo "=== SCP back to local machine ==="
echo "Run this command to copy back (adjust port/IP if needed):"
echo "scp -P $LOCAL_PORT -i ~/.ssh/id_ed25519 -r root@$(hostname -I | awk '{print $1}'):$SUBMISSION_DIR/ ~/Desktop/"
echo ""
echo "Or from your LOCAL machine:"
echo "scp -P <POD_PORT> -i ~/.ssh/id_ed25519 -r root@<POD_IP>:/workspace/parameter-golf/final_model.int6.ptz ~/projects/parameter-golf/"
