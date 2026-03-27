"""
The almost-fully atomic way to train and run inference for DINOv3 in pure, dependency-free Python.
Inspired by Karpathy's microGPT.py
This file is the complete algorithm.
Efficiency is applied for further educational value and runnable speed (improved autograd with matrixes, fused modules)
"""

import os
import math
import random
import gzip
import struct
import copy

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
images = [Raw([[b / 255.0 for b in raw[16 + i * rows * cols + r * cols : 16 + i * rows * cols + (r + 1) * cols]] for r in range(rows)]) for i in range(n)]
print(f"num images: {len(images)}, size: {rows}x{cols}")

#basic arithmetic operators that work across data and matrices
class Arithmetic:                                                                                                
      def __neg__(self): return self * -1                                                                             
      def __radd__(self, other): return self + other                                                                    
      def __sub__(self, other): return self + (-other)                                                                
      def __rsub__(self, other): return other + (-self)
      def __rmul__(self, other): return self * other
      def __truediv__(self, other): return self * other**-1
      def __rtruediv__(self, other): return other * self**-1  

# holding raw data that can be manipulated, somewhat more similar to Pytorch holding raw data and autograd
class Raw(Arithmetic):
    __slots__ = ('data',)
    def __init__(self, data): self.data = data
    @classmethod
    def vals_like(cls, rows, cols, val=0): return cls([[val] * cols for _ in range(rows)])   
    @classmethod 
    def random_init(cls, rows, cols, std=.08): return cls([[random.gauss(0, std) for _ in range(cols)] for _ in range(rows)])
    def shape(self): return (len(self.data), len(self.data[0]) if self.data else 0)
    def __add__(self, other): 
        if isinstance(other, Raw): return Raw([[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(self.data, other.data)])
        else: return Raw([[a + other for a in row] for row in self.data])            
    def __matmul__(self, other): 
        oT = list(zip(*other.data))                                                                                       
        return Raw([[sum(a*b for a,b in zip(row, col)) for col in oT] for row in self.data])
    def __pow__(self, val): return Raw([[a**val for a in row] for row in self.data])
    def __iadd__(self, other): # element in-place addition
        if isinstance(other, Raw):
          for i in range(len(self.data)):
              for j in range(len(self.data[0])):
                  self.data[i][j] += other.data[i][j]
        return self
    def __mul__(self, other):
        if isinstance(other, Raw): return Raw([[a * b for a, b in zip(ra, rb)] for ra, rb in zip(self.data, other.data)]) # hadamard multiplication
        else: return Raw([[a * other for a in row] for row in self.data])
    def __getitem__(self, idx):
      if not isinstance(idx, tuple): idx = (idx, slice(None))
      rows = self.data[idx[0]] if isinstance(idx[0], slice) else [self.data[idx[0]]]
      if isinstance(idx[1], slice): return Raw([row[idx[1]] for row in rows])
      else: return Raw([[row[idx[1]]] for row in rows])
    def acc_at(self, rows, cols, other):  # scatter accumulate for slice backward
      for i, r in enumerate(rows):
          for j, c in enumerate(cols):
              self.data[r][c] += other.data[i][j]
    def T(self): return Raw([list(col) for col in zip(*self.data)])
    def sum_rows(self): return Raw([[sum(row)] * len(row) for row in self.data])
    def clamp(self, lo=0, hi=1): return Raw([[max(lo, min(hi, a)) for a in row] for row in self.data])
    @staticmethod
    def concat(*datas, axis=0):
        if axis == 0:                                                                                                     
            out = []                                                                                                      
            for d in datas: out.extend(d.data)       
            return Raw(out)                                                                                              
        else:
            return Raw([sum([d.data[i] for d in datas], []) for i in range(len(datas[0].data))])      
        

# Tensor with gradients
class Tensor(Arithmetic):
    __slots__ = ('data', 'grad', '_children', '_backward', 'requires_grad')

    def __init__(self, data, children=(), _backward=None, requires_grad=True):
        self.data = data if isinstance(data, Raw) else Raw(data)
        self.grad = Raw.vals_like(*self.data.shape()) if requires_grad else None
        self._children = children
        self._backward = _backward
        self.requires_grad = requires_grad

    @classmethod
    def random_init(cls, rows, cols, std=.08, requires_grad=True):
        return cls(Raw.random_init(rows, cols, std), requires_grad=requires_grad)

    def shape(self): return self.data.shape()
    def _ones(self): return Raw.vals_like(*self.data.shape(), val=1)
    def _apply(self, fn): return Raw([[fn(a) for a in row] for row in self.data.data])

    # unary op: out = f(self), backward = out.grad * local_derivative
    def _unary(self, result, local_grad):
        out = Tensor(result, (self,))
        def _backward():
            if self.grad: self.grad += out.grad * local_grad
        out._backward = _backward
        return out

    def __add__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data + other.data, (self, other))
            def _backward():
                if self.grad:  self.grad += out.grad
                if other.grad: other.grad += out.grad
            out._backward = _backward
            return out
        else:
            out = Tensor(self.data + other, (self,))
            def _backward():
                if self.grad: self.grad += out.grad
            out._backward = _backward
            return out

    def __mul__(self, other):
        if isinstance(other, Tensor): # hadamard multiplication
            out = Tensor(self.data * other.data, (self, other))
            def _backward():
                if self.grad:  self.grad += out.grad * other.data
                if other.grad: other.grad += out.grad * self.data
            out._backward = _backward
            return out
        else: return self._unary(self.data * other, other)  # scalar multiply

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))
        def _backward():
            if self.grad:  self.grad += out.grad @ other.data.T()
            if other.grad: other.grad += self.data.T() @ out.grad
        out._backward = _backward
        return out
    def __pow__(self, n):   return self._unary(self.data ** n, self.data ** (n - 1) * n)
    def log(self):          return self._unary(self._apply(math.log), self.data ** -1)
    def exp(self):          e = self._apply(math.exp); return self._unary(e, e)
    def tanh(self):         t = self._apply(math.tanh); return self._unary(t, self._ones() - t * t)
    def silu(self):
        sig = self._apply(lambda a: 1.0 / (1.0 + math.exp(-a)))
        return self._unary(self.data * sig, sig + self.data * sig * (self._ones() - sig)) # fused silu for speed

    def GELU(self): return 0.5 * self * (1 + (math.sqrt(2 / math.pi) * (self + 0.044715 * self**3)).tanh())

    def row_sum(self):
        out = Tensor(self.data.sum_rows(), (self,))
        def _backward():
            if self.grad:  Raw([[sum(row)] * len(row) for row in out.grad.data])
        out._backward = _backward
        return out

    def sum_all(self):
        total = sum(sum(row) for row in self.data.data)
        out = Tensor(Raw([[total]]), (self,))
        def _backward():
            if self.grad: self.grad += Raw.vals_like(*self.data.shape(), val=out.grad.data[0][0])
        out._backward = _backward
        return out

    def repeat_rows(self, n):
        # broadcasting fixes, unfortunately could not implement strides like PyTorch without unnecessary complexity
        out = Tensor(Raw([self.data.data[0][:] for _ in range(n)]), (self,))
        def _backward():
            if self.grad:
                for j in range(self.data.shape()[1]):
                    self.grad.data[0][j] += sum(out.grad.data[i][j] for i in range(n))
        out._backward = _backward
        return out

    def softmax(self):
        # fusing this for speed. 
        # subtract max per row for numerical stability
        maxes = Raw([[max(row)] * len(row) for row in self.data.data])
        e = Raw([[math.exp(a) for a in row] for row in (self.data - maxes).data]) # subtracting maxes as softmax is shift-invariant
        s = e * Raw([[1.0 / sum(row)] * len(row) for row in e.data]) # multiply values by the sum of the rows (e^term)/sum(e^terms)

        out = Tensor(s, (self,))
        def _backward():
            if self.grad:
                self.grad += s * (out.grad - (out.grad * s).sum_rows())
                # strong fusing of the softmax
        out._backward = _backward
        return out

    def T(self):                  
      out = Tensor(self.data.T(), (self,))
      def _backward():
          if self.grad: self.grad += out.grad.T()                                                                
      out._backward = _backward
      return out   

    def __getitem__(self, idx):
        if not isinstance(idx, tuple): idx = (idx, slice(None))
        rows = list(range(self.data.shape()[0])[idx[0]] if isinstance(idx[0], slice) else [idx[0]])
        cols = list(range(self.data.shape()[1])[idx[1]] if isinstance(idx[1], slice) else [idx[1]])
        out = Tensor(self.data[idx], (self,))
        def _backward():
            if self.grad: self.grad.acc_at(rows, cols, out.grad)
        out._backward = _backward
        return out

    @staticmethod
    def cat(*matrices, axis=0):
        out = Tensor(Raw.concat(*[m.data for m in matrices], axis=axis), tuple(matrices))
        def _backward():
            pos = 0
            for m in matrices:
                s = m.data.shape()[axis]
                if m.grad:
                    if axis == 0: m.grad += out.grad[pos:pos+s]
                    else:         m.grad += out.grad[:, pos:pos+s]
                pos += s
        out._backward = _backward
        return out

    def backward(self):
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
        self.grad = Raw.vals_like(*self.data.shape(), val=1)
        for v in reversed(topo):
            v._backward()

# Augmentations are vital for DINOv3, but are very customizeable
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
    h, w = image.shape()
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    return image[y:y+size, x:x+size]

def blur(image, kernel_size=5, sigma=(0.1, 2.0)):
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sigma_val = random.uniform(sigma[0], sigma[1]) if isinstance(sigma, (tuple, list)) else sigma

    # Build normalized 2D Gaussian kernel.
    ry, rx = kh // 2, kw // 2
    kernel = []
    for y in range(-ry, ry + 1):
        row = []
        for x in range(-rx, rx + 1):
            val = math.exp(-((x * x) / (2.0 * sigma_val * sigma_val) + (y * y) / (2.0 * sigma_val * sigma_val)))
            row.append(val)
        kernel.append(row)
    ksum = sum(sum(row) for row in kernel)
    kernel = [[v / ksum for v in row] for row in kernel]

    h, w = image.shape()
    d = image.data
    out = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    iy = min(max(y + ky - ry, 0), h - 1)
                    ix = min(max(x + kx - rx, 0), w - 1)
                    acc += d[iy][ix] * kernel[ky][kx]
            out[y][x] = acc
    return Raw(out)

def brightness_jitter(image):
    h, w = image.shape()
    return (image * Raw([[random.uniform(0.5, 1.5) for _ in range(w)] for _ in range(h)])).clamp(0, 1)

def gaussian_noise(image, mean=0, std=0.1):
    h, w = image.shape()
    return (image + Raw([[random.gauss(mean, std) for _ in range(w)] for _ in range(h)])).clamp(0, 1)

def augment_image(image, local=False):
    if local:
        for func, prob in [(blur, LOCAL_BLUR_PROB), (brightness_jitter, LOCAL_BRIGHTNESS_JITTER_PROB), (gaussian_noise, LOCAL_GAUSSIAN_NOISE_PROB)]:
            if random.random() < prob:
                image = func(image)
    else:
        for func, prob in [(blur, GLOBAL_BLUR_PROB), (brightness_jitter, GLOBAL_BRIGHTNESS_JITTER_PROB), (gaussian_noise, GLOBAL_GAUSSIAN_NOISE_PROB)]:
            if random.random() < prob:
                image = func(image)
    return image

def get_crops(image):
    global_crops = [crop(augment_image(image), GLOBAL_CROP_SIZE) for _ in range(GLOBAL_CROPS)]
    local_crops = [crop(augment_image(image, local=True), LOCAL_CROP_SIZE) for _ in range(LOCAL_CROPS)]
    return global_crops, local_crops

PATCH_SIZE = 4
N_EMBED = 16
N_LAYER = 1
N_HEAD = 4
HEAD_DIM = N_EMBED // N_HEAD

HEAD_PROTOTYPES = 10 # 10 for 10 classes!
# we simplify the heads to just be linear projections instead of an MLP
student_state_dict = {
    'patch_embed': Tensor.random_init(PATCH_SIZE * PATCH_SIZE, N_EMBED), # patch embedding weights, square size to embed dim
    'CLS_token': Tensor(Raw.random_init(1, N_EMBED, std=0.02)),         # learned CLS token, no registers needed at this scale
    'DINO_head': Tensor.random_init(N_EMBED, HEAD_PROTOTYPES),          # projects CLS embed to prototype dim for DINO loss
    'iBOT_head': Tensor.random_init(N_EMBED, HEAD_PROTOTYPES),          # projects patch embeds to prototype dim for iBOT loss
    # backbone output norm (shared: applied to everything for global crops, patches-only for local crops)
    'norm_gamma': Tensor(Raw.vals_like(1, N_EMBED, val=1)),
    'norm_beta':  Tensor(Raw.vals_like(1, N_EMBED, val=0)),
    # dedicated local CLS norm (paper Sec 3.2: separate norm for local crop CLS tokens during training)
    'local_cls_norm_gamma': Tensor(Raw.vals_like(1, N_EMBED, val=1)),
    'local_cls_norm_beta':  Tensor(Raw.vals_like(1, N_EMBED, val=0)),
}

for i in range(N_LAYER):
    student_state_dict[f'layer{i}.atten_wq'] = Tensor.random_init(N_EMBED, N_EMBED) # attention weight matrices for each layer
    student_state_dict[f'layer{i}.atten_wk'] = Tensor.random_init(N_EMBED, N_EMBED)
    student_state_dict[f'layer{i}.atten_wv'] = Tensor.random_init(N_EMBED, N_EMBED)
    student_state_dict[f'layer{i}.atten_wo'] = Tensor.random_init(N_EMBED, N_EMBED)
    student_state_dict[f'layer{i}.mlp_w1']   = Tensor.random_init(N_EMBED, N_EMBED * 4) # MLP weight matrices, 4x expansion
    student_state_dict[f'layer{i}.mlp_w2']   = Tensor.random_init(N_EMBED * 4, N_EMBED)
    # pre-norm: norm1 before attention, norm2 before MLP (gamma=1, beta=0 init)
    student_state_dict[f'layer{i}.norm1_gamma'] = Tensor(Raw.vals_like(1, N_EMBED, val=1))
    student_state_dict[f'layer{i}.norm1_beta']  = Tensor(Raw.vals_like(1, N_EMBED, val=0))
    student_state_dict[f'layer{i}.norm2_gamma'] = Tensor(Raw.vals_like(1, N_EMBED, val=1))
    student_state_dict[f'layer{i}.norm2_beta']  = Tensor(Raw.vals_like(1, N_EMBED, val=0))
    # layer-scale skipped because only one layer
    #student_state_dict[f'layer{i}.ls'] = 

def compute_rope(H, W, train=True, r=2, head_dim=HEAD_DIM, base=100.0):
    periods = [base ** (2*i / (head_dim//2)) for i in range(head_dim//4)]
    sin_data, cos_data = [], []
    scale = 1
    if train:
        scale = math.exp(random.uniform(-math.log(r), math.log(r)))
    for r in range(H):
        for c in range(W):
            a = [2*math.pi*(2*(r+.5)/H-1)/p * scale for p in periods] + [2*math.pi*(2*(c+.5)/W-1)/p * scale for p in periods]
            a = a * 2  # tile for rotate_half pairing
            sin_data.append([math.sin(x) for x in a])
            cos_data.append([math.cos(x) for x in a])
    return Raw(sin_data), Raw(cos_data)

def rope_rotate_half(x):
    h = x.shape()[1] // 2
    return Tensor.cat(x[:, h:] * -1, x[:, :h], axis=1)

def rope_apply(q, k, sin, cos):
    q_cls, q_patches = q[0:1], q[1:]
    k_cls, k_patches = k[0:1], k[1:]
    q_rot = q_patches * cos + rope_rotate_half(q_patches) * sin
    k_rot = k_patches * cos + rope_rotate_half(k_patches) * sin
    return Tensor.cat(q_cls, q_rot), Tensor.cat(k_cls, k_rot)

def layernorm(x, gamma, beta):
    seq = x.shape()[0]
    mean = (x.row_sum()) * (1.0 / x.shape()[1])
    diff = x + mean * -1
    var = (diff * diff).row_sum() * (1.0 / x.shape()[1])
    normed = diff * (var + 1e-6) ** -0.5
    return normed * gamma.repeat_rows(seq) + beta.repeat_rows(seq) # have to repeat rows to match size

def l2_norm(x):
    return x * (x * x).row_sum() ** -0.5

def vit(image, state_dict, train=True, is_local=False):
    h, w = image.shape()
    H, W = h // PATCH_SIZE, w // PATCH_SIZE

    x = [state_dict['CLS_token']]
    for i in range(H):
        for j in range(W):
            patch = image[i * PATCH_SIZE:(i + 1) * PATCH_SIZE, j * PATCH_SIZE:(j + 1) * PATCH_SIZE]
            x.append(Tensor([sum(patch.data, [])]) @ state_dict['patch_embed'])
            
    x = Tensor.cat(x)
    sin, cos = compute_rope(H, W, train)
    for li in range(N_LAYER):
        x_residual = x
        x = layernorm(x, state_dict[f'layer{li}.norm1_gamma'], state_dict[f'layer{li}.norm1_beta'])
        
        # Attention, it's all we need
        Q = x @ state_dict[f'layer{li}.atten_wq'] # (seq, 16)
        K = x @ state_dict[f'layer{li}.atten_wk']
        V = x @ state_dict[f'layer{li}.atten_wv']

        heads = []
        for h in range(N_HEAD):
            hs = h * HEAD_DIM
            q_h = Q[:, hs:hs+HEAD_DIM]   # (seq, 4)
            k_h = K[:, hs:hs+HEAD_DIM]                                                                       
            v_h = V[:, hs:hs+HEAD_DIM]
            q_h, k_h = rope_apply(q_h, k_h, sin, cos)    
            attn = (q_h @ k_h.T()) * (1.0 / HEAD_DIM**0.5)  # (seq, seq)                                     
            attn = attn.softmax()                                                                            
            heads.append(attn @ v_h)

        x = Tensor.cat(*heads, axis=1)    # (seq, 16)
        x = x @ state_dict[f'layer{li}.atten_wo']
        x = x + x_residual

        x_residual = x
        x = layernorm(x, state_dict[f'layer{li}.norm2_gamma'], state_dict[f'layer{li}.norm2_beta'])
        x = (x @ state_dict[f'layer{li}.mlp_w1']).GELU() # (seq, 64)
        x = x @ state_dict[f'layer{li}.mlp_w2'] # (seq, 16)
        x = x + x_residual

    if train and is_local:
        # Sec 3.2: local crops get dedicated norm for CLS, shared norm for patches
        cls_normed = layernorm(x[0:1], state_dict['local_cls_norm_gamma'], state_dict['local_cls_norm_beta'])
        patch_normed = layernorm(x[1:], state_dict['norm_gamma'], state_dict['norm_beta'])
        x = Tensor.cat(cls_normed, patch_normed)
    else:
        x = layernorm(x, state_dict['norm_gamma'], state_dict['norm_beta'])

    if train:
        cls_out = l2_norm(x[0:1] @ state_dict['DINO_head']) # normally would be MLP but skip for simplicity
        patch_out = l2_norm(x[1:] @ state_dict['iBOT_head'])
        return cls_out, patch_out
    else:
        return x[0:1], x[1:]  # raw CLS and patch embeddings
print(f'Parameters: {sum(x.shape()[0] * x.shape()[1] for x in student_state_dict.values())}')

teacher_state_dict = copy.deepcopy(student_state_dict) # teacher is initialized from student

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [Raw.vals_like(x.shape()[0], x.shape()[1]) for x in student_state_dict.values()] # first moment buffer
v = [Raw.vals_like(x.shape()[0], x.shape()[1]) for x in student_state_dict.values()] # second moment buffer

num_steps = 1000
for step in range(num_steps):
    img = images[step % len(images)]
    global_crops, local_crops = get_crops(img)
    student_embed = vit(img, student_state_dict)
    


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