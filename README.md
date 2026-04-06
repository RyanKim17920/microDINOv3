# nanoDINOv3

DINOv3, recreated in pure, dependency-free Python.

Raw holds the data, Tensor holds the autograd systems.

DINOv2 Sinkhorn-Knopp requires both a large number of prototypes and batch size to be stable (or converges to a uniform distribution), so we include DINO's EMA-centering for teacher predictions as well (testing uses that).
Gram anchoring is incorporated as well, but is unused.

All other aspects of DINOv3 are used.


output.txt is a sample output, with some large time gaps (I closed the computer at times)
