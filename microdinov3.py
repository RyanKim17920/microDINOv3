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
MNIST = "https://ossci-datasets.s3.amazonaws.com/mnist"

def fetch(url):
    try: return gzip.decompress(urllib.request.urlopen(url).read())
    except:
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        return gzip.decompress(urllib.request.urlopen(url, context=ctx).read())

def parse_images(buf):
    _, n, rows, cols = struct.unpack('>IIII', buf[:16])
    return [([[b / 255.0 for b in buf[16 + i*rows*cols + r*cols : 16 + i*rows*cols + (r+1)*cols]]
            for r in range(rows)]) for i in range(n)], rows, cols

def parse_labels(buf):
    _, n = struct.unpack('>II', buf[:8])
    return [buf[8 + i] for i in range(n)]

train_images, rows, cols = parse_images(fetch(f"{MNIST}/train-images-idx3-ubyte.gz"))
train_labels = parse_labels(fetch(f"{MNIST}/train-labels-idx1-ubyte.gz"))
test_images, _, _  = parse_images(fetch(f"{MNIST}/t10k-images-idx3-ubyte.gz"))
test_labels  = parse_labels(fetch(f"{MNIST}/t10k-labels-idx1-ubyte.gz"))
print(f"train: {len(train_images)}, test: {len(test_images)}, size: {rows}x{cols}")

#basic arithmetic operators that work across data and matrices
class Arithmetic:                                                                                                
      def __neg__(self): return self * -1                                                                             
      def __radd__(self, other): return self + other                                                                    
      def __sub__(self, other): return self + (-other)                                                                
      def __rsub__(self, other): return other + (-self)
      def __rmul__(self, other): return self * other
      def __truediv__(self, other): return self * other**-1
      def __rtruediv__(self, other): return other * self**-1  

# holding raw data that can be manipulated, incorporates all math aspects
class Raw(Arithmetic):
    __slots__ = ('data',)
    # construction
    def __init__(self, data): self.data = data
    @classmethod
    def vals_like(cls, rows, cols, val=0): return cls([[val] * cols for _ in range(rows)])
    @classmethod
    def random_init(cls, rows, cols, std=.02): return cls([[random.gauss(0, std) for _ in range(cols)] for _ in range(rows)])
    # shape / info
    def shape(self): return (len(self.data), len(self.data[0]) if self.data else 0)
    def flatten(self): return Raw([sum(self.data, [])])
    # element-wise
    def _apply(self, f): return Raw([[f(x) for x in row] for row in self.data])
    def exp(self):  return self._apply(math.exp)
    def log(self):  return self._apply(math.log)
    def tanh(self): return self._apply(math.tanh)
    def clamp(self, lo=0, hi=1): return Raw([[max(lo, min(hi, a)) for a in row] for row in self.data])
    # arithmetic operators
    def __add__(self, other):
        if isinstance(other, Raw): return Raw([[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(self.data, other.data)])
        else: return Raw([[a + other for a in row] for row in self.data])
    def __iadd__(self, other): # in-place for grad accumulation speed
        if isinstance(other, Raw):
          for i in range(len(self.data)):
              for j in range(len(self.data[0])):
                  self.data[i][j] += other.data[i][j]
        return self
    def __mul__(self, other):
        if isinstance(other, Raw): return Raw([[a * b for a, b in zip(ra, rb)] for ra, rb in zip(self.data, other.data)]) # hadamard multiplication
        else: return Raw([[a * other for a in row] for row in self.data])
    def __matmul__(self, other):
        oT = list(zip(*other.data))
        return Raw([[sum(a*b for a,b in zip(row, col)) for col in oT] for row in self.data])
    def __pow__(self, val): return Raw([[a**val for a in row] for row in self.data])
    # indexing
    def __getitem__(self, idx):
      if not isinstance(idx, tuple): idx = (idx, slice(None))
      rows = self.data[idx[0]] if isinstance(idx[0], slice) else [self.data[idx[0]]]
      if isinstance(idx[1], slice): return Raw([row[idx[1]] for row in rows])
      else: return Raw([[row[idx[1]]] for row in rows])
    def acc_at(self, rows, cols, other):  # scatter accumulate for slice backward
      for i, r in enumerate(rows):
          for j, c in enumerate(cols):
              self.data[r][c] += other.data[i][j]
    # reductions
    def row_sum(self): return Raw([[sum(row)] * len(row) for row in self.data])
    def row_max(self): return Raw([[max(row)] * len(row) for row in self.data])
    def cols_sum(self):
        cols = len(self.data[0]) if self.data else 0
        col_sums = [sum(self.data[r][c] for r in range(len(self.data))) for c in range(cols)]
        return Raw([col_sums[:] for _ in self.data])
    def sum_all(self): return sum(sum(row) for row in self.data)
    # shape transforms
    def T(self): return Raw([list(col) for col in zip(*self.data)])
    def repeat_rows(self, n): return Raw([self.data[0][:] for _ in range(n)])
    @staticmethod
    def concat(*datas, axis=0):
        if axis == 0:
            out = []
            for d in datas: out.extend(d.data)
            return Raw(out)
        else:
            return Raw([sum([d.data[i] for d in datas], []) for i in range(len(datas[0].data))])
    # compound ops
    def softmax(self):
        e = (self - self.row_max()).exp()
        return e * e.row_sum() ** -1      
        

# Autograd engine, raw ops with backward 
class Tensor(Arithmetic):
    __slots__ = ('data', 'grad', '_children', '_backward', 'requires_grad')
    # construction
    def __init__(self, data, children=(), _backward=None, requires_grad=True):
        self.data = data if isinstance(data, Raw) else Raw(data)
        self.grad = Raw.vals_like(*self.data.shape()) if requires_grad else None
        self._children = children
        self._backward = _backward
        self.requires_grad = requires_grad

    @classmethod
    def random_init(cls, rows, cols, std=.02, requires_grad=True):
        return cls(Raw.random_init(rows, cols, std), requires_grad=requires_grad)

    def shape(self): return self.data.shape()

    # autograd helper — handles simple unary ops where backward = out.grad * local_derivative
    def _unary(self, result, local_grad):
        out = Tensor(result, (self,))
        def _backward():
            if self.grad: self.grad += out.grad * local_grad
        out._backward = _backward
        return out

    # arithmetic operators
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
    def __pow__(self, n): return self._unary(self.data ** n, self.data ** (n - 1) * n)

    # element-wise math
    def log(self):  return self._unary(self.data.log(), self.data ** -1)
    def exp(self):  e = self.data.exp(); return self._unary(e, e)
    def tanh(self): t = self.data.tanh(); return self._unary(t, Raw.vals_like(*self.data.shape(), val=1) - t * t)

    # activations
    def GELU(self): return 0.5 * self * (1 + (math.sqrt(2 / math.pi) * (self + 0.044715 * self**3)).tanh())

    # reductions
    def row_sum(self):
        out = Tensor(self.data.row_sum(), (self,))
        def _backward():
            if self.grad: self.grad += out.grad.row_sum()
        out._backward = _backward
        return out

    def sum_all(self):
        total = self.data.sum_all()
        out = Tensor(Raw([[total]]), (self,))
        def _backward():
            if self.grad: self.grad += Raw.vals_like(*self.data.shape(), val=out.grad.data[0][0])
        out._backward = _backward
        return out

    # shape transforms
    def T(self):
        out = Tensor(self.data.T(), (self,))
        def _backward():
            if self.grad: self.grad += out.grad.T()
        out._backward = _backward
        return out

    def repeat_rows(self, n):
        out = Tensor(self.data.repeat_rows(n), (self,))
        def _backward():
            if self.grad:
                for j in range(self.data.shape()[1]):
                    self.grad.data[0][j] += sum(out.grad.data[i][j] for i in range(n))
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

    # compound ops
    def softmax(self):
        s = self.data.softmax()
        out = Tensor(s, (self,))
        def _backward():
            if self.grad:
                self.grad += s * (out.grad - (out.grad * s).row_sum())
        out._backward = _backward
        return out

    # autograd engine
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
            if v._backward: v._backward()

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
            val = math.exp(-((x * x + y * y) / (2.0 * sigma_val * sigma_val)))
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

N_REGISTERS = 1
DROP_PATH_PROB = 0 # set up to 0.3 if multi-layer
LAYER_SCALE = False


HEAD_PROTOTYPES = 10 # 10 for 10 classes!
# we simplify the heads to just be linear projections instead of an MLP
student_state_dict = {
    # patch embedding weights, square size to embed dim
    'patch_embed': Tensor.random_init(PATCH_SIZE * PATCH_SIZE, N_EMBED), 
    # technically could have std = uniform(-1/patch_size, 1/patch_size) but things break down when small scale so keeping =.02
    # learnable tokens
    'CLS_token': Tensor(Raw.random_init(1, N_EMBED)),         # learned CLS token
    'register_tokens':Tensor(Raw.random_init(N_REGISTERS, N_EMBED)), # register
    'mask_token': Tensor(Raw.vals_like(1, N_EMBED, val=0)), # for iBOT
    # projection heads
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
    # layer-scale skipped because only one layer, but put here for fun
    if LAYER_SCALE:
        student_state_dict[f'layer{i}.ls1'] = Tensor(Raw.vals_like(1, N_EMBED, val=1e-5))
        student_state_dict[f'layer{i}.ls2'] = Tensor(Raw.vals_like(1, N_EMBED, val=1e-5))

def compute_rope(H, W, train=True, scale_factor=2, head_dim=HEAD_DIM, base=100.0):
    periods = [base ** (2*i / (head_dim//2)) for i in range(head_dim//4)]
    sin_data, cos_data = [], []
    scale = 1
    if train:
        scale = math.exp(random.uniform(-math.log(scale_factor), math.log(scale_factor)))
    for r in range(H):
        for c in range(W):
            a = [2*math.pi*(2*(r+.5)/H-1)/p * scale for p in periods] + [2*math.pi*(2*(c+.5)/W-1)/p * scale for p in periods]
            a = a * 2  # tile for rotate_half pairing
            sin_data.append([math.sin(x) for x in a])
            cos_data.append([math.cos(x) for x in a])
    return Raw(sin_data), Raw(cos_data)

def rope_rotate_half(x):
    h = x.shape()[1] // 2
    return Tensor.cat(-x[:, h:],  x[:, :h], axis=1)

def rope_apply(q, k, sin, cos, pre_tokens = 1):
    q_cls_register, q_patches = q[0:pre_tokens], q[pre_tokens:]
    k_cls_register, k_patches = k[0:pre_tokens], k[pre_tokens:]
    q_rot = q_patches * cos + rope_rotate_half(q_patches) * sin
    k_rot = k_patches * cos + rope_rotate_half(k_patches) * sin
    return Tensor.cat(q_cls_register, q_rot), Tensor.cat(k_cls_register, k_rot)

def layernorm(x, gamma, beta):
    seq = x.shape()[0]
    mean = (x.row_sum()) * (1.0 / x.shape()[1])
    diff = x - mean
    var = (diff * diff).row_sum() * (1.0 / x.shape()[1])
    normed = diff * (var + 1e-6) ** -0.5
    return normed * gamma.repeat_rows(seq) + beta.repeat_rows(seq) # have to repeat rows to match size

def l2_norm(x):
    return x * ((x * x ).row_sum() + 1e-6) ** -0.5

def vit(image, state_dict, train=True, is_local=False, mask=None):
    h, w = image.shape()
    H, W = h // PATCH_SIZE, w // PATCH_SIZE

    pre_token_count = 1 + N_REGISTERS

    x = [state_dict['CLS_token'], state_dict['register_tokens']]
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if mask is not None and mask[idx]:
                x.append(state_dict['mask_token'])
            else:
                patch = image[i * PATCH_SIZE:(i + 1) * PATCH_SIZE, j * PATCH_SIZE:(j + 1) * PATCH_SIZE]
                x.append(Tensor(patch.flatten()) @ state_dict['patch_embed'])
            
    x = Tensor.cat(*x)
    sin, cos = compute_rope(H, W, train)
    for li in range(N_LAYER):
        x_residual = x
        if random.random() > DROP_PATH_PROB:
            x = layernorm(x, state_dict[f'layer{li}.norm1_gamma'], state_dict[f'layer{li}.norm1_beta'])
            
            # Attention, it's all we need
            Q = x @ state_dict[f'layer{li}.atten_wq'] # (seq, patch^2)
            K = x @ state_dict[f'layer{li}.atten_wk']
            V = x @ state_dict[f'layer{li}.atten_wv']

            heads = []
            for h in range(N_HEAD):
                hs = h * HEAD_DIM
                q_h = Q[:, hs:hs+HEAD_DIM]   # (seq, head_dim)
                k_h = K[:, hs:hs+HEAD_DIM]                                                                       
                v_h = V[:, hs:hs+HEAD_DIM]
                q_h, k_h = rope_apply(q_h, k_h, sin, cos, pre_tokens = pre_token_count)    
                attn = (q_h @ k_h.T()) * (1.0 / HEAD_DIM**0.5)  # (seq, head_dim) @ (head_dim, seq) -> (seq, seq)                                     
                attn = attn.softmax()                                                                            
                heads.append(attn @ v_h)

            x = Tensor.cat(*heads, axis=1)    # (seq, 16) 
            x = x @ state_dict[f'layer{li}.atten_wo'] 
            if LAYER_SCALE:
                x = state_dict[f'layer{li}.ls1'].repeat_rows(x.shape()[0]) * x + x_residual
            else:
                x = x + x_residual

            
            x_residual = x
            x = layernorm(x, state_dict[f'layer{li}.norm2_gamma'], state_dict[f'layer{li}.norm2_beta'])
            x = (x @ state_dict[f'layer{li}.mlp_w1']).GELU() # (seq, 64)
            x = x @ state_dict[f'layer{li}.mlp_w2'] # (seq, 16)
            if LAYER_SCALE:
                x = state_dict[f'layer{li}.ls2'].repeat_rows(x.shape()[0]) * x + x_residual
            else:
                x = x + x_residual

    if train and is_local:
        # Sec 3.2: local crops get dedicated norm for CLS, shared norm for patches
        cls_normed = layernorm(x[0:pre_token_count], state_dict['local_cls_norm_gamma'], state_dict['local_cls_norm_beta'])
        patch_normed = layernorm(x[pre_token_count:], state_dict['norm_gamma'], state_dict['norm_beta'])
        x = Tensor.cat(cls_normed, patch_normed)
    else:
        x = layernorm(x, state_dict['norm_gamma'], state_dict['norm_beta'])
    if train:
        # real DINOv3 head: MLP(in→2048→256) → L2norm → Linear(256→K, no bias), L2 between MLP and final layer
        # makes it cosine similarity to learned prototypes. we skip the MLP, MNIST doesn't need that capacity
        cls_out = l2_norm(x[0:1] @ state_dict['DINO_head'])
        patch_out = l2_norm(x[pre_token_count:] @ state_dict['iBOT_head'])
        return cls_out, patch_out, x[0:1], x[1:pre_token_count], x[pre_token_count:] # DINO, iBot, then normal CLS, registers, patch embeddingss
    else:
        return x[0:1], x[1:pre_token_count], x[pre_token_count:]  # raw CLS, registers, and patch embeddings

teacher_state_dict = copy.deepcopy(student_state_dict) # teacher is initialized from student
for p in teacher_state_dict.values():
    p.requires_grad = False
    p.grad = None

student_params = list(student_state_dict.values())
teacher_params = list(teacher_state_dict.values())

print(f'Parameters: {sum(p.shape()[0] * p.shape()[1] for p in student_params)}')

# training hyperparameters
LEARNING_RATE, BETA1, BETA2, EPS_ADAM = 0.01, 0.85, 0.99, 1e-8
STUDENT_TEMP    = 0.1
TEACHER_TEMP    = 0.04
EMA_MOMENTUM    = 0.996
MASK_RATIO      = 0.25
DINO_WEIGHT     = 1.0
IBOT_WEIGHT     = 1.0
KOLEO_WEIGHT    = 0.1
GRAM_ANCHORING  = True
GRAM_WEIGHT     = 0.1
GRAM_START_STEP = 500
SK_ITERS        = 1 # 3 in DINOv3 but unnecessary compute, 1 approximately works
NUM_STEPS       = 1000
BATCH_SIZE      = 4 # needed for KoLeo loss

# adam buffers
adam_m = [Raw.vals_like(p.shape()[0], p.shape()[1]) for p in student_params] # first moment
adam_v = [Raw.vals_like(p.shape()[0], p.shape()[1]) for p in student_params] # second moment

def random_mask(num_patches, ratio=MASK_RATIO):
    return [random.random() < ratio for _ in range(num_patches)]

def koleo_loss(cls_tensors):
    # KoLeo is the -mean(log(nearest_neighbor_distance)) so requires more than 2 CLS tokens 
    B = len(cls_tensors)
    if B < 2:
        return Tensor(Raw([[0.0]]))
    normed = [l2_norm(c) for c in cls_tensors]
    raw_vecs = [n.data.data[0] for n in normed]
    loss = Tensor(Raw([[0.0]]))
    for i in range(B):
        best_j, best_dot = -1, -float('inf')
        for j in range(B):
            if i == j: continue
            dot = sum(raw_vecs[i][k] * raw_vecs[j][k] for k in range(N_EMBED))
            if dot > best_dot:
                best_dot, best_j = dot, j
        diff = normed[i] - normed[best_j]
        dist = ((diff * diff).sum_all() + 1e-8) ** 0.5
        loss = loss - dist.log()
    return loss * (1.0 / B)

def gram_loss(student_feats, teacher_feats):
    # gram loss: ||G_student - G_teacher||^2 where G = X^T @ X / n
    # for anchoring feature correlations
    n, d = student_feats.shape()
    s_gram = student_feats.T() @ student_feats * (1.0 / n)
    t_gram = teacher_feats.T() @ teacher_feats * (1.0 / n)
    diff = s_gram - t_gram
    return (diff * diff).sum_all() * (1.0 / (d * d))

def sinkhorn_knopp(logits_raw, iters=SK_ITERS):
    # converts logits_raw into a doubly-stochastic matrix over iterations, replacing EMA centering 
    q = logits_raw.exp() if isinstance(logits_raw, Raw) else logits_raw.data.exp()
    for _ in range(iters):
        q /= q.row_sum()   # row-normalize
        q /= q.cols_sum()  # col-normalize
    return q

def log_softmax_tensor(x, temp=STUDENT_TEMP):
    # for numerical stability, log_softmax: x/temp - max - log(sum(exp(x/temp - max)))
    scaled = x * (1.0 / temp)
    shifted = scaled - scaled.data.row_max()  # subtract per-row max (Raw, no grad through max)
    return shifted - shifted.exp().row_sum().log()


for step in range(NUM_STEPS):
    total_loss = Tensor(Raw([[0.0]]))
    koleo_cls_tokens = []

    for b in range(BATCH_SIZE):
        img = Raw(train_images[(step * BATCH_SIZE + b) % len(train_images)])
        global_crops, local_crops = get_crops(img)

        # precompute: per-crop masks, teacher outputs, SK probs
        n_gp = (GLOBAL_CROP_SIZE // PATCH_SIZE) ** 2
        masks = [random_mask(n_gp) for _ in global_crops]
        masked_idxs = [[i for i, m in enumerate(mask) if m] for mask in masks]

        teacher_outs = [vit(gc, teacher_state_dict, train=True) for gc in global_crops]
        teacher_dino_probs = [sinkhorn_knopp(t[0].data * (1.0 / TEACHER_TEMP)) for t in teacher_outs]
        teacher_ibot_probs = [sinkhorn_knopp(Raw([t[1].data.data[i] for i in mi]) * (1.0 / TEACHER_TEMP))
                              if mi else None for t, mi in zip(teacher_outs, masked_idxs)]
        teacher_patch_feats = [t[4].data for t in teacher_outs]

        # student global crops 
        for gi, gc in enumerate(global_crops):
            s_dino, s_ibot, s_cls_pre, s_regs, s_patch_pre = vit(gc, student_state_dict, train=True, mask=masks[gi])
            koleo_cls_tokens.append(s_cls_pre)

            # DINO CLS: student global vs each other teacher global 
            for ti, t_dino_prob in enumerate(teacher_dino_probs):
                if gi == ti: continue
                total_loss -= (log_softmax_tensor(s_dino) * t_dino_prob).sum_all() * DINO_WEIGHT

            # iBOT: student masked patch head outputs vs teacher masked patch probs
            if masked_idxs[gi] and teacher_ibot_probs[gi] is not None:
                s_masked = Tensor.cat(*[s_ibot[i:i+1] for i in masked_idxs[gi]])
                total_loss -= (log_softmax_tensor(s_masked) * teacher_ibot_probs[gi]).sum_all() * (IBOT_WEIGHT / len(masked_idxs[gi]))

            # gram anchoring (pre-head patch features, after warmup)
            if GRAM_ANCHORING and step >= GRAM_START_STEP:
                total_loss += gram_loss(s_patch_pre, teacher_patch_feats[gi]) * GRAM_WEIGHT

        # student local crops (DINO CLS only)
        for lc in local_crops:
            s_dino_local = vit(lc, student_state_dict, train=True, is_local=True)[0]
            for t_dino_prob in teacher_dino_probs:
                total_loss -= (log_softmax_tensor(s_dino_local) * t_dino_prob).sum_all() * DINO_WEIGHT

    # KoLeo across all student global CLS tokens in batch
    if len(koleo_cls_tokens) >= 2:
        total_loss += koleo_loss(koleo_cls_tokens) * KOLEO_WEIGHT

    total_loss /= BATCH_SIZE # backward the loss

    total_loss.backward()

    # adam + ema teacher + zero grad
    bc1, bc2 = 1 - BETA1 ** (step + 1), 1 - BETA2 ** (step + 1)
    for i, (sp, tp) in enumerate(zip(student_params, teacher_params)):
        if sp.grad is None: continue
        adam_m[i] = adam_m[i] * BETA1 + sp.grad * (1 - BETA1)
        adam_v[i] = adam_v[i] * BETA2 + (sp.grad * sp.grad) * (1 - BETA2)
        m_hat = adam_m[i] * (1.0 / bc1)
        v_hat = adam_v[i] * (1.0 / bc2)
        # convert to raw to get rid of backprop
        sp.data -= LEARNING_RATE * m_hat / (v_hat ** 0.5 + EPS_ADAM)
        tp.data = tp.data * EMA_MOMENTUM + sp.data * (1 - EMA_MOMENTUM)
        sp.grad = Raw.vals_like(*sp.shape())

    # snapshot gram anchor teacher at warmup boundary (DINOv3 had specific point and updated over iters but too complex for this)
    if GRAM_ANCHORING and step == GRAM_START_STEP:
        gram_state_dict = copy.deepcopy(teacher_state_dict)

    if step % 50 == 0:
        print(f"step {step}, loss: {total_loss.data.data[0][0]:.4f}")

print("\n--- KNN Evaluation ---")
KNN_IMAGES = 500
TOP_K  = 5

embeddings = []
# embed 500 random train images
for i in range(KNN_IMAGES):
    cls, _, _ = vit(Raw(train_images[i]), student_state_dict, train=False)
    vec = cls.data
    embeddings.append(l2_norm(vec).data[0])

# query 10 random test images, find 5 nearest ones by cosine similarity
for _ in range(10):
    qi = random.randint(0, len(test_images) - 1)                              
    cls, _, _ = vit(Raw(test_images[qi]), student_state_dict, train=False)
    qv = l2_norm(cls.data).data[0]
    sims = [sum(a*b for a, b in zip(qv, emb)) for emb in embeddings]
    top_k = sorted(range(KNN_IMAGES), key=lambda i: sims[i], reverse=True)[:TOP_K]                                                             
    print(f"  [{test_labels[qi]}]: {[train_labels[i] for i in top_k]}")  