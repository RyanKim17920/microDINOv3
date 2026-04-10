# nanoDINOv3

DINOv3 reimplemented in ~800 lines of pure, dependency-free Python.
No torch, no numpy — just stdlib. Inspired by Karpathy's microGPT.

## What's implemented
- Custom autograd engine (`Raw` + `Tensor` with backward pass)
- Vision Transformer with Rotary Position Embeddings (RoPE)
- DINO + iBOT dual-head self-supervised objectives
- EMA centering (stable) and optional Sinkhorn-Knopp
- KoLeo regularization, register tokens, optional layer scale
- KNN evaluation on MNIST

## Run
```
python microdinov3.py
```
Downloads MNIST automatically (~60MB). Runs on CPU.

## Results (27k params, 2000 steps, CPU)
| Eval | Accuracy | Baseline |
|---|---|---|
| KNN on pre-head CLS (32-dim) | 35.2% | 10% random |
| KNN on DINO head output (32-dim) | 29.2% | 10% random |

`output.txt` contains a full training log.

## Notes
DINOv2's Sinkhorn-Knopp centering needs large batches and many prototypes to stay stable (otherwise it collapses to uniform). At this scale, DINOv1-style EMA centering is more reliable, so both are implemented and EMA is default. Gram anchoring is scaffolded but unused.

## References
- [DINOv3: Self-supervised learning for vision at scale](https://arxiv.org/abs/2508.10104)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [Emerging Properties in Self-Supervised Vision Transformers (original DINO)](https://arxiv.org/abs/2104.14294)
- [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
