#!/bin/bash
set -e
echo "=== Parameter Golf Setup ==="

# Clone Avery's fork on the competition branch
cd /workspace
git clone https://github.com/bentenavery/parameter-golf.git
cd parameter-golf
git checkout avery/diffattn-ttt
echo "Branch: $(git branch --show-current) | Commit: $(git log --oneline -1)"

# Install deps
pip install sentencepiece huggingface-hub datasets tqdm numpy torch zstandard --quiet

# Download FineWeb dataset (1024 vocab, 10 train shards)
# Set HF_TOKEN as env var before running: export HF_TOKEN=your_token
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

echo "=== Setup complete. Ready to train. ==="
echo "Run: bash /workspace/parameter-golf/run_competition.sh"
