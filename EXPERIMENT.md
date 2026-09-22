# Experiment: Shape vs Texture Bias in CNN

## Hypothesis

Standard CNNs trained on image classification learn to rely on **texture cues** more than **shape cues** when making predictions — following the finding of Geirhos et al. (2019) on ImageNet-trained models. This experiment replicates and probes that bias using STL-10.

---

## Reference

Geirhos, R. et al. (2019). *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.* ICLR 2019.
https://github.com/rgeirhos/texture-vs-shape

---

## Implementation Status

| Step | Script | Status |
|---|---|---|
| Train VGG on STL-10 | `train.py` | ✅ Complete |
| Generate cue-conflict images | `generate_conflicts.py` | ✅ Complete |
| Evaluate shape vs texture bias | `evaluate.py` | ✅ Complete |

**Run order:**
```bash
python train.py                 # trains VGG → checkpoints/best.pth
python generate_conflicts.py    # generates 4,500 conflict images → conflicts/
python evaluate.py              # measures bias → results/
```

---

## Experiment Design

### Step 1 — Train a CNN on STL-10
**Script:** `train.py`

- Train a CNN from scratch on STL-10 (5,000 training images, 10 classes, 96×96).
- STL-10 chosen over CIFAR-10 (32×32) so that style-transferred cue-conflict images retain visible shape and texture at the resolution the CNN sees.
- **Architecture: VGG-style** (small 3×3 conv filters only, channels double after each max-pool, no skip connections)
  ```
  Input (96×96×3)
  → Conv 3×3 → 64 → ReLU
  → Conv 3×3 → 64 → ReLU → MaxPool 2×2       (→ 48×48)
  → Conv 3×3 → 128 → ReLU
  → Conv 3×3 → 128 → ReLU → MaxPool 2×2      (→ 24×24)
  → Conv 3×3 → 256 → ReLU → MaxPool 2×2      (→ 12×12)
  → Flatten
  → FC 512 → ReLU → Dropout
  → FC 10 → Softmax
  ```
- No pretrained weights — trained from scratch.
- Saves best checkpoint to `checkpoints/best.pth`.

### Step 2 — Generate Cue-Conflict Images
**Script:** `generate_conflicts.py`

- Takes STL-10 test images as source material.
- For every ordered class pair (A, B) where A ≠ B, generates 50 conflict images:
  - **Shape** comes from a class-A image (content/structure preserved via VGG19 relu4_2)
  - **Texture** comes from a class-B image (style transferred via Gram matrix loss on relu1_1 through relu5_1)
- Method: Gatys et al. neural style transfer, 300 LBFGS steps, pretrained VGG19 feature extractor.
- Total: 90 pairs × 50 images = **4,500 conflict images** saved to `conflicts/<shape>_<texture>/NNN.png`.
- Resume-friendly: skips already-completed pairs.

### Step 3 — Evaluate Shape vs Texture Bias
**Script:** `evaluate.py`

- Loads trained VGG from `checkpoints/best.pth`.
- Runs model on all conflict images.
- For each image, records whether the prediction matches:
  - Class A (shape donor) → **shape-biased decision**
  - Class B (texture donor) → **texture-biased decision**
  - Neither → tracked separately, excluded from bias ratio

### Step 4 — Measure and Visualise
**Script:** `evaluate.py` (outputs to `results/`)

- **Shape bias score** = shape-correct / (shape-correct + texture-correct)
  - Score near 1.0 → strongly shape-biased
  - Score near 0.0 → strongly texture-biased
  - Score near 0.5 → no preference
- `results/results.csv` — per-image predictions
- `results/bias_scores.csv` — per-pair shape bias scores
- `results/heatmap.png` — 10×10 heatmap (green = shape-biased, red = texture-biased)
- `results/summary.txt` — overall score + top 5 most shape/texture-biased pairs

---

## Expected Outcome

Based on Geirhos et al., we expect the CNN to show **texture bias** — predicting the texture class more often than the shape class on conflict images. A shape bias score well below 0.5 would confirm this.

---

## Decisions

- [x] Dataset → STL-10 at 96×96 (CIFAR-10 at 32×32 too small for style transfer)
- [x] Architecture → VGG-style, single model, trained from scratch
- [x] Framework → PyTorch
- [x] Conflict pairs → all 90 ordered pairs, 50 images each
- [x] Style transfer → Gatys et al. LBFGS, pretrained VGG19 feature extractor
