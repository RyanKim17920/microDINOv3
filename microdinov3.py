"""
The almost-most atomic way to train and run inference for DINOv3 in pure, dependency-free Python.
Inspired by Karpathy's microGPT.py
This file is the complete algorithm.
Everything else is just efficiency (well I did apply some efficiency to convert Value to Matrix so it can actually run).
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
images = [[[b / 255.0 for b in raw[16 + i * rows * cols + r * cols : 16 + i * rows * cols + (r + 1) * cols]] for r in range(rows)] for i in range(n)]
print(f"num images: {len(images)}, size: {rows}x{cols}")

class Matrix:
    __slots__ = ('data', 'grad', '_children', '_backward', 'requires_grad')

    def __init__(self, data, children=(), _backward=None, requires_grad = True):
        self.data = data                                                   # list of scalar values of this node calculated during forward pass MxN shape
        self.grad = self.vals_like(self, 0) if requires_grad else None     # list of derivatives of the loss w.r.t. this node, calculated in backward pass
        self._children = children                                          # children of this node in the computation graph
        self._backward = _backward                                         # backward function for gradient computation
        self.requires_grad = requires_grad

    def __add__(self, other):
        if isinstance(other, Matrix):
            assert self.shape() == other.shape() # elementwise addition for same shape matrices
            out = Matrix(
                [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(self.data, other.data)],
                (self, other),
            )
            out._backward = lambda: Matrix._elementwise_backward(out, (self, other), lambda g, a, b: (g, g)) 
            return out
        else:
            out = Matrix(
                [[a + other for a in row] for row in self.data],
                (self,),
            )
            out._backward = lambda: Matrix._elementwise_backward(out, (self,), lambda g, a: (g,)) 
            return out
    
    def __mul__(self, other):
        if isinstance(other, Matrix):
            assert self.shape()[1] == other.shape()[0] # NxP for self, PxM for other
            def _backward():
                # dL/dself = dL/dout * dout/dself = out.grad @ other.T
                # dL/dother = dL/dout * dout/dother = self.T @ out.grad
                if self.grad is not None:
                    self.grad = [[sg + sum(g * b for g, b in zip(out_row, col_b))                                                                                    
                                    for sg, col_b in zip(sg_row, zip(*other.data))]                                                                                  
                                        for sg_row, out_row in zip(self.grad, out.grad)]
                if other.grad is not None: 
                    other.grad = [[og + sum(a * g for a, g in zip(col_a, col_g))                                                                                     
                                for og, col_g in zip(og_row, zip(*out.grad))]                                                                                     
                                        for og_row, col_a in zip(other.grad, zip(*self.data))]      
            out = Matrix(
                # multiply rowA by colB for each rowA in self and colB in other and sum to get each element of the output
                [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in zip(*other.data)] for row_a in self.data],
                (self, other),
            )
            out._backward = _backward
            return out
        else:
            out = Matrix(
                [[a * other for a in row] for row in self.data],
                (self,),
            )
            out._backward = lambda: Matrix._elementwise_backward(out, (self,), lambda g, a: (g * other,))
            # scalar multiplication just scales the gradient by other
            return out  

    def __pow__(self, other): 
        out = Matrix([[a ** other for a in row] for row in self.data],(self,),)
        out._backward = lambda: Matrix._elementwise_backward(out, (self,), lambda g, a: (g * other * a ** (other - 1),)) 
        # power rule: d(a^b)/da = b * a^(b-1) -- gradient scaled by other and a^(other-1)
        return out
    
    def log(self):
        out = Matrix([[math.log(a) for a in row] for row in self.data],(self,),)
        out._backward = lambda: Matrix._elementwise_backward(out, (self,), lambda g, a: (g / a,)) 
        # derivative of log(a) is 1/a -- gradient scaled by 1/a
        return out

    def exp(self):
        out = Matrix([[math.exp(a) for a in row] for row in self.data],(self,),)
        out._backward = lambda: Matrix._elementwise_backward(out, (self,), lambda g, a: (g * math.exp(a),)) 
        # derivative of exp(a) is exp(a) -- gradient scaled by exp(a)
        return out

    def silu(self):
        # x times sigmoid(x) or x / (1 + exp(-x))
        out = Matrix([[a / (1 + math.exp(-a)) for a in row] for row in self.data],(self,),)
        out._backward = lambda: Matrix._elementwise_backward(out, (self,), lambda g, a: (g * (1 / (1 + math.exp(-a)) + a * math.exp(-a) / ((1 + math.exp(-a)) ** 2)),)) 
        # derivative of silu(a) is sigmoid(a) + a * sigmoid'(a) -- gradient scaled by that
        # sigmoid'(a) = sigmoid(a) * (1 - sigmoid(a))
        # yes I fused SiLU, but you could do it separately if needed (autograd handles)
        return out
    
    def tanh(self):
        out = Matrix([[math.tanh(a) for a in row] for row in self.data],(self,),)
        out._backward = lambda: Matrix._elementwise_backward(out, (self,), lambda g, a: (g * (1 - math.tanh(a) ** 2),)) 
        # derivative of tanh(a) is 1 - tanh(a)^2 -- gradient scaled by that
        # some more fusing because faster for GELU
        return out

    def GELU(self):
        return 0.5 * self * (1 + (math.sqrt(2 / math.pi) * (self + 0.044715 * self**3)).tanh()) # this is the exact GELU formula, but the derivative is a mess so not fusing it

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    
    def vals_like(self, matrix, val):
        return [([val] * len(matrix.data[0])) for _ in range(len(matrix.data))]
    
    def shape(self):
        return (len(self.data), len(self.data[0]) if self.data else 0)
    
    @staticmethod
    def _elementwise_backward(out, children, grad_fn):
      for i in range(len(out.data)):
          for j in range(len(out.data[0])):
              g = out.grad[i][j]
              vals = tuple(c.data[i][j] for c in children)
              grads = grad_fn(g, *vals)
              for child, cg in zip(children, grads):
                  if child.grad is not None:
                    child.grad[i][j] += cg
    def backward(self):
        # non-recursive to prevent any potential issues
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            v, done = stack.pop()
            if done: topo.append(v)
            elif id(v) not in visited and v.requires_grad:
                visited.add(id(v))
                stack.append((v, True))
                for c in v._children: stack.append((c, False))
        self.grad = self.vals_like(self, 1)
        for i in reversed(topo):
            i._backward()



# Augmentations are vital for DINOv3
GLOBAL_CROP_SIZE              = 24  # DINOv3 uses 224 global out of 256, but we're using MNIST
LOCAL_CROP_SIZE               = 16  # DINOv3 uses 96 local out of 256
GLOBAL_CROPS                  = 1   # used 2 global crops
LOCAL_CROPS                   = 2   # used 8 local crops

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


PATCH_SIZE = 4
N_EMBED = 16
N_LAYER = 1
N_HEAD = 4
HEAD_DIM = N_EMBED // N_HEAD


"""
# TODO:
    - IMAGE PROCESSING:
        - DINOv3 MNIST loaded [x]
        - Augmentations (random crops, blur) [X]
        - Local/Global Crops [X]
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