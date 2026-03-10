"""
The most atomic way to train and run inference for DINOv3 in pure, dependency-free Python.
Inspired by Karpathy's microGPT.py
This file is the complete algorithm.
Everything else is just efficiency.
"""

import os
import math
import random
import gzip
import struct

random.seed(42)

# Let there be a Dataset of images. We will use the MNIST dataset of handwritten digits, which is small and easy to work with.
import urllib.request, ssl            
url = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"                                                                 
try: data = urllib.request.urlopen(url).read()                                                                                                 
except:                      
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    data = urllib.request.urlopen(url, context=ctx).read()
raw = gzip.decompress(data)

# parse the raw MNIST data into a list of images that are normalized
_, n, rows, cols = struct.unpack('>IIII', raw[:16])
images = [[b / 255.0 for b in raw[16 + i*rows*cols : 16 + (i+1)*rows*cols]] for i in range(n)]
print(f"num images: {len(images)}, size: {rows}x{cols}")

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad




"""
# TODO:
    - IMAGE PROCESSING:
        - DINOv3 MNIST loaded [x]
        - Augmentations (random crops, flips [ ]
        - Local/Global Crops [ ]
    - Autograd implemented [x]
    - MODEL:
        - 4x4 Patch embedding [ ]
        - CLS token [ ]
        - MHSA [ ]
        - RoPE [ ]
        - RMSNorm [ ]
        - MLP block [ ]
        - L2-normalized output projection heads [ ]
    - DINO CORE:
        - Student model (Value weights, recieves gradients) [ ]
        - Teacher model (float weights, no gradients) [ ]
        - EMA teacher update (momentum ~.0996) [ ]
        - 
        - Model output sharpening [ ]
        - DINOv3 CLS loss [ ]
        - iBot loss [ ]
        - Gram Anchoring loss [ ]
        - KoLeo Loss [ ]
    - INFERENCE (the payoff):
        - Embed 100 test images with frozen student [ ]
        - For each, find 5 nearest neighbors by CLS cosine similarity [ ]
        - Print digit label vs neighbor labels [ ]
"""