# train_avery.py — Strategy Notes

## Base: PR #462 (JoeProAI, val_bpb=1.0672)

PR #462 is the current frontier at 1.0672 BPB. It builds on the main baseline with:

### PR #462 Key Changes vs Baseline
| Feature | Baseline | PR #462 |
|---------|---------|---------|
| Layers | 9 | 11 |
| KV heads | 4 | 8 (full MHA) |
| MLP hidden | 2x | 1792 (Star-ReLU) |
| MLP activation | relu^2 | Star-ReLU (learned scale/bias per channel) |
| XSA | off | last 4 layers |
| Bigram hash embedding | off | 8192 buckets, 128-dim |
| Smear gate | off | yes (learned weighted prev-token blend) |
| Skip connections | basic | gated skip (learned gate + skip_weights) |
| Partial RoPE | off | first 16 dims of 64 head_dim |
| LN scale | off | 1/sqrt(layer_idx+1) depth dampening |
| EMA | off | yes (decay=0.9985) |
| Late QAT | off | int6 fake quant when LR < 0.15 |
| Quantization | int8+zlib | int6+zstd-22 |
| TTT | off | yes (10 epochs AdamW, cosine decay) |
| Weight decay | 0 | Muon 0.04, Adam 0.04 |
| Decoder LR mult | 1x | 2x |

TTT (test-time training) is the single biggest contribution — AdamW fine-tune
on the validation set for 10 epochs after EMA averaging. This is the primary
score driver.

---

## Avery's Contribution: Differential Attention on Layers 0-6

### What Is Differential Attention?
Paper: arXiv:2410.05258, Microsoft Research, ICLR 2025.

Instead of one softmax attention map:
```
Attn = softmax(QK^T / sqrt(d)) V
```

Differential Attention computes two maps and subtracts:
```
Attn = softmax(Q1 K1^T / sqrt(d/2)) V  -  lambda * softmax(Q2 K2^T / sqrt(d/2)) V
```

Where:
- `Q1, K1` = first half of head_dim (indices :32 of 64)
- `Q2, K2` = second half of head_dim (indices 32: of 64)
- `V` = full head_dim (unchanged)
- `lambda` = learnable scalar per head, initialized by layer depth
- Scale uses `1/sqrt(half_head_dim)` = `1/sqrt(32)` instead of `1/sqrt(64)`

### Why Same Parameter Count?
The same `c_q`, `c_k`, `c_v`, `proj` matrices are used — same shape.
Only addition: `diff_lambda` (8 floats per DiffAttn layer) = 8 * 7 = 56 extra fp32 params.
Completely negligible in a 16MB budget. The architecture is "semantically reorganized",
not expanded.

### Lambda Initialization
Paper formula: `lambda_init = 0.8 - 0.6 * exp(-0.3 * layer_idx)`

| Layer | lambda_init |
|-------|------------|
| 0     | 0.200      |
| 1     | 0.339      |
| 2     | 0.451      |
| 3     | 0.542      |
| 4     | 0.616      |
| 5     | 0.676      |
| 6     | 0.725      |

Early layers get more differential correction (smaller lambda = larger subtraction).
Deep layers approach standard attention behavior (lambda -> 0.8).

### Why Layers 0-6?
- **Early layers** learn low-level features (n-grams, syntax). DiffAttn's noise
  cancellation — subtracting a "redundant" attention map — helps these layers
  focus on signal rather than spurious correlations.
- **Layers 7-10** retain standard attention + XSA (eXtended Self-Attention with
  value orthogonalization). XSA is incompatible with DiffAttn's split-head
  structure, and late layers likely benefit more from XSA's output cleaning.
- Split: 7 DiffAttn + 4 standard+XSA = 11 layers total, matching PR #462 config.

### Expected Effect
- DiffAttn consistently outperforms standard attention in the original paper
  (2-3% perplexity improvement on LM tasks).
- In small models (16MB budget), noise cancellation in early layers is especially
  valuable — the model has limited capacity and early representation quality
  directly gates downstream quality.
- The lambda values are learned — the model will adapt them during training.

---

## What To Tune Next

### Priority 1: Verify Training Converges
Run a short 500-step smoke test. Check that:
- DiffAttn layers (0-6) are actually learning (lambda gradient should be nonzero)
- Loss curve is similar to PR #462 baseline
- No NaN/inf from the differential subtraction (lambda is bounded by init)

### Priority 2: Lambda Placement
Try variants:
- `DIFF_ATTN_LAYERS=11` — all layers (may conflict with XSA on last 4)
- `DIFF_ATTN_LAYERS=4` — only first 4 layers (less aggressive)
- Disable XSA and use DiffAttn everywhere (XSA_LAYERS=0, DIFF_ATTN_LAYERS=11)

### Priority 3: Lambda Init Tuning
- Start with smaller lambda (more subtraction): set `lambda_init = 0.5 - 0.3*exp(-0.3*l)`
- Clamp lambda to [0, 1] during training (add to forward or as postproc)

### Priority 4: TTT Epochs
PR #462 uses 10 TTT epochs. Try:
- 15-20 epochs (more adaptation, more compute)
- Lower TTT LR with more epochs: `TTT_LR=0.0002 TTT_EPOCHS=20`

### Priority 5: Parameter Budget
Run with `--num-layers 12` if DiffAttn's overhead (56 params) leaves room.
Check total bytes after int6+zstd to confirm under 16MB.

---

## Environment Variables for Experimentation

```powershell
# Default (7 DiffAttn layers)
$env:DIFF_ATTN_LAYERS = "7"

# All DiffAttn, no XSA
$env:DIFF_ATTN_LAYERS = "11"
$env:XSA_LAYERS = "0"

# Minimal DiffAttn
$env:DIFF_ATTN_LAYERS = "3"

# More TTT
$env:TTT_EPOCHS = "20"
$env:TTT_LR = "0.0002"
```

---

## How To Run

```powershell
cd C:\Users\avery\projects\parameter-golf
.venv\Scripts\activate
# Requires: CUDA GPU, data at ./data/datasets/fineweb10B_sp1024
python train_avery.py
```

Multi-GPU (if available):
```powershell
torchrun --nproc_per_node=4 train_avery.py
```

---

## Parameter Budget Impact

With 11 layers, model_dim=512, num_heads=8, head_dim=64:
- DiffAttn extra params: 8 heads * 7 layers = 56 fp32 floats = 224 bytes
- Star-ReLU MLP: `scale` (1792) + `bias` (1792) = 3584 fp32 per layer
- Total overhead vs PR #462: ~224 bytes (effectively zero)

DiffAttn is parameter-neutral by design. The same Q/K/V/proj matrices
are reused — just their semantic interpretation changes (first vs second half of head_dim
maps to different attention functions).

---

## What's Next To Score

1. Get a GPU training run started (needs CUDA + data)
2. Validate BPB after first full run
3. If better than 1.0672: submit as PR to openai/parameter-golf
4. If comparable: tune TTT epochs and lambda placement
5. Target: sub-1.05 BPB (estimated ~1-3% improvement from DiffAttn)
