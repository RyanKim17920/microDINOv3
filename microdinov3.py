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


# Augmentations are vital for DINOv3
GLOBAL_CROP_SIZE              = 20  # DINOv3 uses 224 global out of 256, but we're using MNIST
LOCAL_CROP_SIZE               = 10  # DINOv3 uses 96 local out of 256
GLOBAL_CROPS                  = 2
LOCAL_CROPS                   = 4   # Often 8 local crops

GLOBAL_BLUR_PROB              = 0.1
LOCAL_BLUR_PROB               = 0.5

GLOBAL_BRIGHTNESS_JITTER_PROB = 0.3
LOCAL_BRIGHTNESS_JITTER_PROB  = 0.8

GLOBAL_GAUSSIAN_NOISE_PROB    = 0.1
LOCAL_GAUSSIAN_NOISE_PROB     = 0.3

def crop(image, size):
    # Randomly crop a square of the given size from the image
    x = random.randint(0, len(image[0]) - size)
    y = random.randint(0, len(image) - size)
    return [row[x:x+size] for row in image[y:y+size]]

def blur(image, kernel_size=5, sigma=(0.1, 2.0)):
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sigma_val = random.uniform(sigma[0], sigma[1]) if isinstance(sigma, (tuple, list)) else sigma
    sigma_x = sigma_y = sigma_val

    # Build normalized 2D Gaussian kernel.
    ry, rx = kh // 2, kw // 2
    kernel = []
    for y in range(-ry, ry + 1):
        row = []
        for x in range(-rx, rx + 1):
            val = math.exp(-((x * x) / (2.0 * sigma_x * sigma_x) + (y * y) / (2.0 * sigma_y * sigma_y)))
            row.append(val)
        kernel.append(row)
    ksum = sum(sum(row) for row in kernel)
    kernel = [[v / ksum for v in row] for row in kernel]

    # Convolve with border clamping.
    h = len(image)
    w = len(image[0])
    out = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    iy = min(max(y + ky - ry, 0), h - 1)
                    ix = min(max(x + kx - rx, 0), w - 1)
                    acc += image[iy][ix] * kernel[ky][kx]
            out[y][x] = acc
    return out

def brightness_jitter(image):
    factor = random.uniform(0.5, 1.5)  # Random brightness factor
    return [[min(max(pixel * factor, 0), 1) for pixel in row] for row in image]

def gaussian_noise(image, mean=0, std=0.1):
    return [[min(max(pixel + random.gauss(mean, std), 0), 1) for pixel in row] for row in image]

def augment_image(image):
    for func, prob in [(blur, GLOBAL_BLUR_PROB), (brightness_jitter, GLOBAL_BRIGHTNESS_JITTER_PROB), (gaussian_noise, GLOBAL_GAUSSIAN_NOISE_PROB)]:
        if random.random() < prob:
            image = func(image)
    return image

def get_crops(image):
    global_crops = [crop(augment_image(image), GLOBAL_CROP_SIZE) for _ in range(GLOBAL_CROPS)]
    local_crops = [crop(augment_image(image), LOCAL_CROP_SIZE) for _ in range(LOCAL_CROPS)]
    return global_crops, local_crops

"""
# TODO:
    - IMAGE PROCESSING:
        - DINOv3 MNIST loaded [x]
        - Augmentations (random crops, blur) [ ]
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