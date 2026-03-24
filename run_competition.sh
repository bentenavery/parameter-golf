#!/bin/bash
set -e
cd /workspace/parameter-golf

echo "=== Starting 3-seed run on 8xH100 ==="
echo "Branch: $(git branch --show-current) | Commit: $(git log --oneline -1)"
echo ""

# Seed 1
echo "=== SEED 1337 ===" | tee -a train_seed1337.log
SEED=1337 torchrun --nproc_per_node=8 train_avery.py 2>&1 | tee -a train_seed1337.log

# Seed 2
echo "=== SEED 42 ===" | tee -a train_seed42.log
SEED=42 torchrun --nproc_per_node=8 train_avery.py 2>&1 | tee -a train_seed42.log

# Seed 3
echo "=== SEED 123 ===" | tee -a train_seed123.log
SEED=123 torchrun --nproc_per_node=8 train_avery.py 2>&1 | tee -a train_seed123.log

echo ""
echo "=== All seeds complete ==="
echo "=== val_bpb summary ==="
grep "final_int6" train_seed*.log | grep "val_bpb"
