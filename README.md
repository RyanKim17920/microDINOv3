# microDINOv3

A DINOv3-style self-supervised ViT in ~850 lines of pure, dependency-free Python.
No torch, no numpy — just stdlib. Inspired by Karpathy's microGPT.

The point of this repo is the algorithm, written out in full: a reverse-mode autograd
engine, a Vision Transformer, and the complete self-distillation training loop, with
nothing hidden behind a framework.

## What's implemented

**Autograd** — matrix-valued reverse-mode automatic differentiation (`Raw` holds data,
`Tensor` adds the tape). Backward passes for matmul, slicing/concatenation, softmax,
LayerNorm (differentiating through both the mean and the variance), and the rest.

**Backbone** — ViT with axial RoPE, register tokens, pre-norm blocks, optional LayerScale,
and a dedicated CLS LayerNorm for local crops. RoPE follows the paper: periods
`base^(2i/(d/2))` with `base=100`, coordinates normalized to `[-1,1]` per axis, and the
RoPE-box jitter augmentation (a single log-uniform rescale in `[1/2, 2]`, applied during
training only).

**Objectives**
- DINO global loss on CLS tokens, with 2 global + 2 local crops and the paper's
  cross-view routing (a crop is never matched against a teacher target from itself).
- iBOT masked patch-level loss, global crops only.
- KoLeo regularization on pre-head CLS tokens.
- **Gram anchoring** — DINOv3's core contribution. Anchors the `N x N` matrix of pairwise
  cosine similarities between *L2-normalized* patch features to a frozen earlier snapshot
  of the teacher, refreshed a bounded number of times. It constrains the geometry of
  patch-to-patch relationships rather than the features themselves, which is what keeps
  dense features from degrading over long training runs.
- Teacher targets via EMA centering (default) or Sinkhorn-Knopp.

**Heads** — untied DINO and iBOT projection heads, each `MLP -> L2 normalize the
bottleneck -> linear to prototypes`. The L2 normalization sits on the bottleneck, not on
the output logits; normalizing the logit vector itself would cap every logit at
`1/sqrt(K)` and make a peaked distribution impossible at temperature 0.1.

**Training** — AdamW with decoupled weight decay, linear LR warmup then constant LR,
global gradient-norm clipping, and a constant teacher EMA momentum. DINOv3 removes
schedules for LR, weight decay, and teacher momentum, and that is followed here.

## Run

```
python microdinov3.py
```

Downloads MNIST automatically (~60MB, cached in `.mnist_cache/`) and runs on CPU.
Python 3.12+ is recommended: the matmul inner loop uses `math.sumprod`, which is a C
builtin added in 3.12. On older versions it falls back to a pure-Python equivalent and
still runs correctly, just slower.
Hyperparameters can be overridden by environment variable for short runs:

```
NUM_STEPS=100 BATCH_SIZE=8 GRAM_START_STEP=50 python microdinov3.py
```

Weights are written to `student_final.json` / `teacher_final.json` at the end.

## Results

2000 steps at batch 32, 9.06h on one CPU core. kNN probe on MNIST with a 5000-image
database, top-5, over the full 10000-image test set. Every figure carries its binomial
standard error.

| Probe | Accuracy |
|---|---|
| Random init (identical weights, zero training) | 29.85% +/- 0.46 |
| **Trained, pre-head CLS (32-dim)** | **59.00% +/- 0.49** |
| Trained, post-head DINO output (32-dim) | 31.33% +/- 0.46 |
| Raw pixels (784-dim), same kNN protocol | 94.42% +/- 0.23 |
| Chance | 10.00% |

**Training delta over the random-init control: +29.15 +/- 0.67 points, 43.4 sigma.**
Self-supervised training roughly doubles the quality of the backbone representation, and
at this sample size that is far outside the noise.

### Learning curve

Probed every 200 steps on a smaller split (1000-image database, 2000 test images), so
these absolute values are **not comparable to the table above**:

| step | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 |
|---|---|---|---|---|---|---|---|---|---|
| acc % | 30.75 | 37.25 | 43.20 | 47.50 | 46.55 | 44.85 | 46.10 | 47.15 | 45.20 |

Standard error is about 1.1 points throughout. The curve climbs steeply to roughly step
800 and then **plateaus** — every point from 800 onward sits within about two standard
errors of the others.

Re-probing the *final* weights under the curve protocol gives **45.15% +/- 1.11**, against
**59.00% +/- 0.49** for those same weights under the final protocol. So the entire
45% -> 59% difference is the larger kNN database, not training, and the final weights are
statistically indistinguishable from step 1800 (45.20%). 2000 steps was more than
sufficient: this model saturates at its capacity (32,736 parameters, 32-dim output), and
more compute would not have helped.

### Which objective did the work

The per-objective losses say something the accuracy numbers alone do not. Against the
uniform-prediction floor of ln(32) = 3.466:

| loss | step 0 | step 1975 |
|---|---|---|
| DINO (global, CLS) | 1.516 | 3.455 |
| iBOT (patch, masked) | 4.601 | 2.491 |
| KoLeo | 0.554 | 0.020 |
| Gram (from step 500) | 0.132 | 0.035 |

**The DINO global objective never left the uniform floor.** It sits at ~3.46 for the whole
run, i.e. the student's CLS distribution stays essentially uniform over the 32 prototypes.
The representation gain is attributable to the patch-level iBOT objective, which drops well
below the floor, together with KoLeo and Gram anchoring.

That also explains the post-head result. The post-head probe reads the DINO head's output,
and that head learned almost nothing — 31.33% against the 29.85% random-init control is
+1.48 +/- 0.65, barely two sigma. Some gap between pre-head and post-head is expected in
any DINO-family model, since the projection head is trained to solve the pretext task and
discards information that a downstream probe would use, which is why the literature probes
pre-head features. Here the gap is far larger than usual because the global objective
driving that head did not train at all at this scale.

`output.txt` contains the full log for this run.

**On baselines.** The meaningful control for a representation probe is *the same
architecture at initialization*, not chance. A randomly initialized ViT already scores
29.85% here, so a "vs 10% random" comparison would badly overstate what training
accomplished; the run evaluates its own untouched starting weights and reports the delta
against those. In the other direction, **raw pixels beat the learned representation
outright, 94.42% to 59.00%** — a 32-dimensional bottleneck throws away information that a
pixel-space nearest-neighbour search keeps, and on a dataset as easy as MNIST that
information matters more than any structure the encoder discovers. Both probes run with
`train=False`, so no RoPE jitter is applied at inference and the numbers are deterministic.

## Scope and honest limitations

This is an educational miniature. The algorithm is implemented and it demonstrably learns
a representation at this scale, but it does **not** reproduce the paper's results and is
not trying to. The paper trains a 6.7B-parameter ViT on 1.69B curated images for 1M
iterations at batch 4096; this trains 32,736 parameters on MNIST for 2000 steps at batch
32. Two consequences are visible in the results above and worth stating plainly: the DINO
global objective does not train at this capacity, and the learned 32-dim representation
loses to raw pixels on this dataset.

Deliberately out of scope:
- **High-resolution adaptation** and **post-hoc distillation** into smaller students
  (the paper's ViT-S/B/L/H+ and ConvNeXt models), including multi-student distillation.
- **Text alignment** (dino.txt).
- The **high-resolution Gram teacher** — the paper runs the Gram teacher at 2x the
  student's resolution and bicubic-downsamples its feature map. Here the Gram teacher sees
  the same resolution as the student.
- **SwiGLU** feed-forward layers; this uses a GELU MLP at 4x expansion.
- **Block-wise masking** for iBOT; masking here is i.i.d. per patch, where the paper masks
  a crop with probability 0.5 and then masks a `[0.1, 0.5]` ratio in blocks.
- Sinkhorn-Knopp is implemented and correct, but EMA centering is the default. The paper
  uses Sinkhorn for both heads; at a batch of 32 with 32 prototypes, SK sits close to the
  uniform solution and centering is better behaved.
- LayerScale, drop-path, and multi-crop counts beyond 2+2 are present but not tuned.

## References
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [Emerging Properties in Self-Supervised Vision Transformers (original DINO)](https://arxiv.org/abs/2104.14294)
- [iBOT](https://arxiv.org/abs/2111.07832)
- [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
