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
Hyperparameters can be overridden by environment variable for short runs:

```
NUM_STEPS=100 BATCH_SIZE=8 GRAM_START_STEP=50 python microdinov3.py
```

Weights are written to `student_final.json` / `teacher_final.json` at the end.

## Results

kNN probe on MNIST (200-image database, top-5, 1000 test images), 150 steps at batch 8.

| Probe | Accuracy |
|---|---|
| Random init (identical weights, zero training) | *see `output.txt`* |
| Trained, pre-head CLS (32-dim) | *see `output.txt`* |
| Trained, post-head DINO output (32-dim) | *see `output.txt`* |

**On baselines.** The meaningful control for a representation probe is *the same
architecture at initialization*, not chance. A randomly initialized ViT already scores far
above 10% on this task, so "vs 10% random" would overstate what training accomplished; the
run therefore evaluates its own starting weights and reports the delta. For scale on the
other side: raw pixels under the identical kNN protocol score around 80%, well above
anything this 32-dim representation produces. Both evaluation probes run with
`train=False`, so no RoPE jitter is applied at inference and the numbers are deterministic.

`output.txt` contains a full training log.

## Scope and honest limitations

This is an educational miniature, not a reproduction of the paper's results. The paper
trains a 6.7B-parameter ViT on 1.69B curated images for 1M iterations at batch 4096. This
trains ~33k parameters on MNIST. The algorithm is implemented; the scale is not, and the
representation quality reflects that.

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
