"""
DINOv3 in pure, dependency-free Python. No torch, no numpy — just stdlib.
This file is the complete algorithm.
Inspired by Karpathy's microGPT.
Autograd operates on matrices (not scalars) and modules are fused — the only concessions to runnable speed.
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

CACHE_DIR = ".mnist_cache"

def fetch(url):
    # cached on disk so repeated runs do not re-download ~60MB every time
    path = os.path.join(CACHE_DIR, url.rsplit('/', 1)[-1])
    if os.path.exists(path):
        with open(path, 'rb') as f: return gzip.decompress(f.read())
    try: raw = urllib.request.urlopen(url).read()
    except:
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        raw = urllib.request.urlopen(url, context=ctx).read()
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, 'wb') as f: f.write(raw)
    return gzip.decompress(raw)

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
    def item(self): return self.data[0][0]
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
    def item(self): return self.data.item()

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
GLOBAL_CROPS                  = 2   # 2 global crops, so the DINO global term is a genuine cross-view pair
LOCAL_CROPS                   = 2   # DINOv3 uses 8

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
N_EMBED = 32
N_LAYER = 2
N_HEAD = 4
HEAD_DIM = N_EMBED // N_HEAD

N_REGISTERS = 1
DROP_PATH_PROB = 0.0 # set up to 0.3 if multi-layer
LAYER_SCALE = False


HEAD_PROTOTYPES = 32 # prototypes != classes, more prototypes lets SK produce peaked targets
HEAD_HIDDEN     = 64 # head MLP hidden width (paper: 8192)
HEAD_BOTTLENECK = 16 # head MLP output width, L2-normalized before the prototype layer (paper: 512 DINO / 384 iBOT)
# Head shape follows the real one: MLP(in -> hidden -> bottleneck) -> L2norm -> Linear(bottleneck -> K, no bias).
# The L2 norm sits on the BOTTLENECK, not on the output logits: normalizing the K-dim logit vector would cap every
# logit at ~1/sqrt(K) and make it impossible for the student to produce a peaked distribution at temperature 0.1.
student_state_dict = {
    # patch embedding weights, square size to embed dim
    'patch_embed': Tensor.random_init(PATCH_SIZE * PATCH_SIZE, N_EMBED),
    # technically could have std = uniform(-1/patch_size, 1/patch_size) but things break down when small scale so keeping =.02
    # learnable tokens
    'CLS_token': Tensor(Raw.random_init(1, N_EMBED)),         # learned CLS token
    'register_tokens':Tensor(Raw.random_init(N_REGISTERS, N_EMBED)), # register
    'mask_token': Tensor(Raw.vals_like(1, N_EMBED, val=0)), # for iBOT
    # projection heads (untied: DINO and iBOT get separate weights, as in the paper)
    # The prototype layer is initialized so each prototype vector has ~unit norm, making the logits
    # genuine cosine similarities in [-1, 1]. The paper's std=0.02 gives norm 0.02*sqrt(512)~0.45 at
    # their 512-d bottleneck; the same std at a 16-d bottleneck gives 0.08, which pins the softmax to
    # uniform at temperature 0.1 no matter what the backbone does.
    'DINO_mlp1': Tensor.random_init(N_EMBED, HEAD_HIDDEN),
    'DINO_mlp2': Tensor.random_init(HEAD_HIDDEN, HEAD_BOTTLENECK),
    'DINO_last': Tensor.random_init(HEAD_BOTTLENECK, HEAD_PROTOTYPES, std=HEAD_BOTTLENECK ** -0.5),
    'iBOT_mlp1': Tensor.random_init(N_EMBED, HEAD_HIDDEN),
    'iBOT_mlp2': Tensor.random_init(HEAD_HIDDEN, HEAD_BOTTLENECK),
    'iBOT_last': Tensor.random_init(HEAD_BOTTLENECK, HEAD_PROTOTYPES, std=HEAD_BOTTLENECK ** -0.5),
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

def projection_head(x, state_dict, prefix):
    # MLP -> L2 normalize the bottleneck -> linear to prototypes (no bias).
    h = (x @ state_dict[f'{prefix}_mlp1']).GELU()
    bottleneck = l2_norm(h @ state_dict[f'{prefix}_mlp2'])
    return bottleneck @ state_dict[f'{prefix}_last']

def vit(image, state_dict, train=True, is_local=False, mask=None, apply_head=None):
    # `train` controls training-only behaviour (RoPE box jitter, the dedicated local-crop CLS norm).
    # `apply_head` controls whether the projection heads run, and defaults to `train` for backwards
    # compatibility. They are separate so evaluation can read head outputs without enabling jitter.
    if apply_head is None:
        apply_head = train
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
    # DINO head out, iBOT head out, raw CLS, registers, patch embeddings.
    # Head outputs are None when apply_head is False; the shape of the tuple never changes.
    cls_out   = projection_head(x[0:1], state_dict, 'DINO') if apply_head else None
    patch_out = projection_head(x[pre_token_count:], state_dict, 'iBOT') if apply_head else None
    return cls_out, patch_out, x[0:1], x[1:pre_token_count], x[pre_token_count:]

teacher_state_dict = copy.deepcopy(student_state_dict) # teacher is initialized from student
for p in teacher_state_dict.values():
    p.requires_grad = False
    p.grad = None

# kept untouched for the whole run so evaluation can report a random-init control against the exact
# weights this run started from. chance (10%) is not a meaningful baseline for a representation probe.
init_state_dict = copy.deepcopy(student_state_dict)
for p in init_state_dict.values():
    p.requires_grad = False
    p.grad = None

student_params = list(student_state_dict.values())
teacher_params = list(teacher_state_dict.values())


# training hyperparameters. env overrides exist so a short smoke run can be launched without editing the file.
def _envf(name, default): return float(os.environ.get(name, default))
def _envi(name, default): return int(os.environ.get(name, default))

LEARNING_RATE, BETA1, BETA2, EPS_ADAM = _envf('LR', 0.001), 0.9, 0.999, 1e-8
WEIGHT_DECAY    = _envf('WEIGHT_DECAY', 0.04)  # AdamW, decoupled, matrices only (paper: constant 0.04)
WARMUP_STEPS    = _envi('WARMUP_STEPS', 100)   # linear LR warmup (paper: 100k iters of a 1M-iter run)
CLIP_GRAD       = _envf('CLIP_GRAD', 30.0)     # global grad-norm clip (paper: 30.0)
STUDENT_TEMP    = 0.1
TEACHER_TEMP    = 0.07
EMA_MOMENTUM    = 0.999
MASK_RATIO      = 0.3
DINO_WEIGHT     = 1.0
IBOT_WEIGHT     = 1.0
KOLEO_WEIGHT    = 0.1
GRAM_ANCHORING  = os.environ.get('GRAM_ANCHORING', '1') == '1'
GRAM_WEIGHT     = _envf('GRAM_WEIGHT', 2.0)    # paper appendix C: w_Gram = 2
GRAM_START_STEP = _envi('GRAM_START_STEP', 500)
GRAM_REFRESH    = _envi('GRAM_REFRESH', 500)   # refresh the Gram teacher every N steps (paper: every 10k iters)
GRAM_MAX_UPDATES= _envi('GRAM_MAX_UPDATES', 3) # paper: at most 3 refreshes
USE_SINKHORN    = os.environ.get('USE_SINKHORN', '0') == '1'
SK_ITERS        = 3
CENTER_MOMENTUM = _envf('CENTER_MOMENTUM', 0.9)  # paper/DINOv2 use 0.9; 0.9999 left the center inert over 2k steps
NUM_STEPS       = _envi('NUM_STEPS', 2000)
BATCH_SIZE      = _envi('BATCH_SIZE', 32)
LOG_EVERY       = _envi('LOG_EVERY', 100)

# DINO multi-crop loss scaling, following ssl_meta_arch.py: each group is averaged over its
# (student crop x teacher crop) pairs, then weighted by that group's share of the total pair count.
DINO_GLOBAL_TERMS = GLOBAL_CROPS * (GLOBAL_CROPS - 1)  # same-crop pairs are skipped
DINO_LOCAL_TERMS  = GLOBAL_CROPS * LOCAL_CROPS
DINO_GLOBAL_SCALE = DINO_GLOBAL_TERMS / (DINO_GLOBAL_TERMS + DINO_LOCAL_TERMS) if DINO_GLOBAL_TERMS else 0.0
DINO_LOCAL_SCALE  = DINO_LOCAL_TERMS / (DINO_GLOBAL_TERMS + DINO_LOCAL_TERMS)
KOLEO_SCALE       = GLOBAL_CROPS

# adam buffers
adam_m = [Raw.vals_like(p.shape()[0], p.shape()[1]) for p in student_params] # first moment
adam_v = [Raw.vals_like(p.shape()[0], p.shape()[1]) for p in student_params] # second moment
if not USE_SINKHORN:
    cls_center = Raw.vals_like(1, HEAD_PROTOTYPES)
    # DINOv1-style EMA centering. DINOv3 uses Sinkhorn-Knopp for both heads (ssl_meta_arch.py asserts it);
    # SK is implemented and correct here, but wants a large batch and many prototypes to stay away from the
    # uniform solution, so EMA centering remains the default at this scale. Set USE_SINKHORN=1 to switch.
    ibot_center = Raw.vals_like(1, HEAD_PROTOTYPES)

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
    # Gram anchoring (paper Eq. 2): L_Gram = || X_S X_S^T - X_G X_G^T ||_F^2, where X is the matrix of
    # *L2-normalized* patch features. So the anchored quantity is the N x N matrix of pairwise patch
    # cosine similarities -- the structure of patch-to-patch relationships -- not the d x d feature
    # covariance. Individual features stay free to move; only their relative geometry is pinned to the
    # Gram teacher. That is what preserves patch locality over long training runs.
    # Reduction is a mean over the N x N entries, matching torch MSELoss in dinov3/loss/gram_loss.py.
    n, _ = student_feats.shape()
    s = l2_norm(student_feats)
    t = l2_norm(teacher_feats)
    diff = (s @ s.T()) - (t @ t.T())
    return (diff * diff).sum_all() * (1.0 / (n * n))

def sinkhorn_knopp(logits_raw, temp=TEACHER_TEMP, iters=SK_ITERS):
    # Sinkhorn-Knopp teacher targets, following dinov3/loss/dino_clstoken_loss.py.
    # Takes (B, K) logits, returns (B, K) assignments whose ROWS sum to 1.
    # Internally Q is K-by-B, matching the paper's notation. The previous implementation ended on a
    # column normalization, so its rows summed to K/B rather than 1 -- correct only by accident when
    # the matrix happened to be square, and wrong for the masked-patch case where B varies per crop.
    scaled = logits_raw * (1.0 / temp)
    gmax = max(max(row) for row in scaled.data)  # global shift: cancels exactly in the sum_Q normalization
    Q = (scaled + (-gmax)).exp().T()             # (K, B)
    K, B = Q.shape()
    Q = Q * (1.0 / (Q.sum_all() + 1e-12))
    for _ in range(iters):
        Q = Q * ((Q.row_sum() + 1e-12) ** -1) * (1.0 / K)   # each prototype takes total weight 1/K
        Q = Q * ((Q.cols_sum() + 1e-12) ** -1) * (1.0 / B)  # each sample takes total weight 1/B
    return (Q * B).T()  # undo the final /B so every sample's row sums to 1

def log_softmax_tensor(x, temp=STUDENT_TEMP):
    # for numerical stability, log_softmax: x/temp - max - log(sum(exp(x/temp - max)))
    scaled = x * (1.0 / temp)
    shifted = scaled - scaled.data.row_max()  # subtract per-row max (Raw, no grad through max)
    return shifted - shifted.exp().row_sum().log()


import time
OUTPUT_PATH = os.getenv('OUTPUT', 'output.txt')

outf = open(OUTPUT_PATH, 'w')
header = (f"N_EMBED={N_EMBED} N_LAYER={N_LAYER} N_HEAD={N_HEAD} HEAD_PROTOTYPES={HEAD_PROTOTYPES} "
          f"BATCH_SIZE={BATCH_SIZE} NUM_STEPS={NUM_STEPS} LR={LEARNING_RATE} STUDENT_TEMP={STUDENT_TEMP} "
          f"TEACHER_TEMP={TEACHER_TEMP} EMA_MOMENTUM={EMA_MOMENTUM} MASK_RATIO={MASK_RATIO} "
          f"DINO_WEIGHT={DINO_WEIGHT} IBOT_WEIGHT={IBOT_WEIGHT} KOLEO_WEIGHT={KOLEO_WEIGHT} "
          f"USE_SINKHORN={USE_SINKHORN} CENTER_MOMENTUM={CENTER_MOMENTUM} GRAM_ANCHORING={GRAM_ANCHORING} "
          f"GRAM_WEIGHT={GRAM_WEIGHT} GRAM_START_STEP={GRAM_START_STEP} GLOBAL_CROPS={GLOBAL_CROPS} "
          f"LOCAL_CROPS={LOCAL_CROPS} WEIGHT_DECAY={WEIGHT_DECAY} WARMUP_STEPS={WARMUP_STEPS} CLIP_GRAD={CLIP_GRAD}")
outf.write(header + '\n')
outf.flush()

def log(msg):
    print(msg, flush=True)
    outf.write(msg + '\n')
    outf.flush()

log(f"train: {len(train_images)}, test: {len(test_images)}, size: {rows}x{cols}")
log(f'Parameters: {sum(p.shape()[0] * p.shape()[1] for p in student_params)}')

t_start = time.time()

# Gram teacher: a frozen snapshot of the EMA teacher, taken once dense features are healthy and then
# refreshed a bounded number of times. It must exist BEFORE the loss that reads it, so the snapshot is
# taken at the top of the step rather than after the optimizer.
gram_state_dict = None
gram_updates = 0

# names that should not get weight decay: norms, learned tokens (torch excludes 1-d params likewise)
NO_DECAY = tuple(k for k in student_state_dict if 'norm' in k or 'token' in k or k.endswith('ls1') or k.endswith('ls2'))
no_decay_flags = [k in NO_DECAY for k in student_state_dict]

for step in range(NUM_STEPS):
    # refresh the Gram teacher at the start of the step, so it is always defined when the loss reads it
    if GRAM_ANCHORING and step >= GRAM_START_STEP:
        due = (gram_state_dict is None) or (GRAM_REFRESH > 0 and (step - GRAM_START_STEP) % GRAM_REFRESH == 0)
        if due and gram_updates <= GRAM_MAX_UPDATES:
            gram_state_dict = copy.deepcopy(teacher_state_dict)
            gram_updates += 1

    total_loss = Tensor(Raw([[0.0]]))
    dino_loss_acc, ibot_loss_acc, koleo_loss_acc, gram_loss_acc = 0.0, 0.0, 0.0, 0.0
    koleo_cls_by_crop = [[] for _ in range(GLOBAL_CROPS)]

    # phase 1: teacher forward for entire batch, collect outputs
    batch_data = []
    n_gp = (GLOBAL_CROP_SIZE // PATCH_SIZE) ** 2
    all_teacher_cls = []
    for b in range(BATCH_SIZE):
        img = Raw(train_images[(step * BATCH_SIZE + b) % len(train_images)])
        global_crops, local_crops = get_crops(img)
        masks = [random_mask(n_gp) for _ in global_crops]
        masked_idxs = [[i for i, m in enumerate(mask) if m] for mask in masks]
        # teacher runs in eval mode: no RoPE box jitter, matching the official teacher being .eval()
        teacher_outs = [vit(gc, teacher_state_dict, train=False, apply_head=True) for gc in global_crops]
        for t in teacher_outs:
            all_teacher_cls.append(t[0].data.data[0])
        batch_data.append((global_crops, local_crops, masks, masked_idxs, teacher_outs))

    # phase 2: teacher CLS targets
    teacher_cls_raw = Raw(all_teacher_cls)
    if USE_SINKHORN:
        teacher_cls_sk = sinkhorn_knopp(teacher_cls_raw)
    else:
        centered = teacher_cls_raw - cls_center.repeat_rows(len(all_teacher_cls))
        teacher_cls_sk = (centered * (1.0 / TEACHER_TEMP)).softmax()
        batch_mean = Raw([[sum(teacher_cls_raw.data[r][c] for r in range(len(all_teacher_cls))) / len(all_teacher_cls)
                           for c in range(HEAD_PROTOTYPES)]])
        cls_center = cls_center * CENTER_MOMENTUM + batch_mean * (1 - CENTER_MOMENTUM)

    # phase 3: student forward + losses
    for b, (global_crops, local_crops, masks, masked_idxs, teacher_outs) in enumerate(batch_data):
        teacher_dino_probs = [Raw([teacher_cls_sk.data[b * GLOBAL_CROPS + gi]]) for gi in range(GLOBAL_CROPS)]
        if USE_SINKHORN:
            teacher_ibot_probs = [sinkhorn_knopp(Raw([t[1].data.data[i] for i in mi]))
                                  if mi else None for t, mi in zip(teacher_outs, masked_idxs)]
        else:
            # DINOv1-style EMA centering for iBOT patch targets (mirrors cls_center pattern above)
            teacher_ibot_probs = []
            for t, mi in zip(teacher_outs, masked_idxs):
                if mi:
                    patch_logits = Raw([t[1].data.data[i] for i in mi])
                    centered = patch_logits - ibot_center.repeat_rows(len(mi))
                    teacher_ibot_probs.append((centered * (1.0 / TEACHER_TEMP)).softmax())
                    # EMA update ibot_center with mean of this crop's masked patch logits
                    patch_mean = Raw([[sum(patch_logits.data[r][c] for r in range(len(mi))) / len(mi)
                                       for c in range(HEAD_PROTOTYPES)]])
                    ibot_center = ibot_center * CENTER_MOMENTUM + patch_mean * (1 - CENTER_MOMENTUM)
                else:
                    teacher_ibot_probs.append(None)

        # DINO terms are averaged within their group, then weighted by the group's share of all pairs
        dino_global_sum, dino_local_sum = Tensor(Raw([[0.0]])), Tensor(Raw([[0.0]]))

        for gi, gc in enumerate(global_crops):
            s_dino, s_ibot, s_cls_pre, s_regs, s_patch_pre = vit(gc, student_state_dict, train=True, mask=masks[gi])
            koleo_cls_by_crop[gi].append(s_cls_pre)

            for ti, t_dino_prob in enumerate(teacher_dino_probs):
                if gi == ti: continue  # same crop: skip, otherwise this is student(x) -> teacher(x) self-prediction
                l = -(log_softmax_tensor(s_dino) * t_dino_prob).sum_all()
                dino_global_sum += l
                dino_loss_acc += l.item()

            if masked_idxs[gi] and teacher_ibot_probs[gi] is not None:
                s_masked = Tensor.cat(*[s_ibot[i:i+1] for i in masked_idxs[gi]])
                l = -(log_softmax_tensor(s_masked) * teacher_ibot_probs[gi]).sum_all() * (1.0 / len(masked_idxs[gi]))
                total_loss += l * IBOT_WEIGHT
                ibot_loss_acc += l.item()

            if gram_state_dict is not None:
                # Gram teacher features are constants: read .data so no gradient flows into the snapshot
                gram_feats = vit(gc, gram_state_dict, train=False, apply_head=False)[4].data
                l = gram_loss(s_patch_pre, gram_feats)
                total_loss += l * GRAM_WEIGHT
                gram_loss_acc += l.item()

        for lc in local_crops:
            s_dino_local = vit(lc, student_state_dict, train=True, is_local=True)[0]
            for t_dino_prob in teacher_dino_probs:
                l = -(log_softmax_tensor(s_dino_local) * t_dino_prob).sum_all()
                dino_local_sum += l
                dino_loss_acc += l.item()

        if DINO_GLOBAL_TERMS:
            total_loss += dino_global_sum * (DINO_WEIGHT * DINO_GLOBAL_SCALE / DINO_GLOBAL_TERMS)
        total_loss += dino_local_sum * (DINO_WEIGHT * DINO_LOCAL_SCALE / DINO_LOCAL_TERMS)

    # per-image losses accumulated above are averaged over the batch
    total_loss /= BATCH_SIZE

    # KoLeo is already a batch-level quantity (it ranks CLS tokens *against each other* across the batch),
    # so it is added after the batch average rather than before -- adding it before divided it by BATCH_SIZE
    # a second time and left it ~32x under-weighted. Per ssl_meta_arch.py it is averaged over global crops
    # and then scaled by the number of global crops.
    koleo_terms = [koleo_loss(toks) for toks in koleo_cls_by_crop if len(toks) >= 2]
    if koleo_terms:
        kl = Tensor(Raw([[0.0]]))
        for t in koleo_terms: kl += t
        kl = kl * (KOLEO_WEIGHT * KOLEO_SCALE / len(koleo_terms))
        total_loss += kl
        koleo_loss_acc = kl.item()

    total_loss.backward()

    # global grad-norm clipping
    sq = sum(sum(v * v for v in row) for p in student_params if p.grad for row in p.grad.data)
    gnorm = sq ** 0.5
    clip_scale = min(1.0, CLIP_GRAD / (gnorm + 1e-6))

    # linear LR warmup, then constant (DINOv3 keeps LR/wd/momentum constant after warmup)
    lr = LEARNING_RATE * min(1.0, (step + 1) / WARMUP_STEPS) if WARMUP_STEPS > 0 else LEARNING_RATE

    # adamw + ema teacher + zero grad
    bc1, bc2 = 1 - BETA1 ** (step + 1), 1 - BETA2 ** (step + 1)
    for i, (sp, tp) in enumerate(zip(student_params, teacher_params)):
        if sp.grad is None: continue
        g = sp.grad * clip_scale
        adam_m[i] = adam_m[i] * BETA1 + g * (1 - BETA1)
        adam_v[i] = adam_v[i] * BETA2 + (g * g) * (1 - BETA2)
        m_hat = adam_m[i] * (1.0 / bc1)
        v_hat = adam_v[i] * (1.0 / bc2)
        update = m_hat / (v_hat ** 0.5 + EPS_ADAM)
        if not no_decay_flags[i]:
            update = update + sp.data * WEIGHT_DECAY  # decoupled weight decay
        sp.data -= update * lr
        tp.data = tp.data * EMA_MOMENTUM + sp.data * (1 - EMA_MOMENTUM)
        sp.grad = Raw.vals_like(*sp.shape())

    if step % LOG_EVERY == 0:
        # Report each objective as a mean per term, so they are directly comparable to the collapse
        # floor ln(HEAD_PROTOTYPES) instead of scaling with the number of crops.
        elapsed = time.time() - t_start
        n_dino_terms = BATCH_SIZE * (DINO_GLOBAL_TERMS + DINO_LOCAL_TERMS)
        n_crop_terms = BATCH_SIZE * GLOBAL_CROPS
        log(f"step {step:4d} | dino: {dino_loss_acc/n_dino_terms:.3f}  ibot: {ibot_loss_acc/n_crop_terms:.3f}  "
            f"koleo: {koleo_loss_acc:.3f}  gram: {gram_loss_acc/n_crop_terms:.4f} | "
            f"total: {total_loss.item():.3f} | lr {lr:.2e} | gnorm {gnorm:.2f} | {elapsed:.1f}s")

KNN_IMAGES = _envi('KNN_IMAGES', 500)
TOP_K = _envi('TOP_K', 5)
N_TEST = _envi('N_TEST', len(test_images))
test_images, test_labels = test_images[:N_TEST], test_labels[:N_TEST]

def knn_evaluate(embeddings, embed_labels, test_imgs, test_lbls, top_k=TOP_K):
    """Evaluate kNN accuracy over all test images. Returns (correct, total, example_predictions)."""
    total = len(test_imgs)
    correct = 0
    examples = []  # collect first 10 for visual inspection
    for qi in range(total):
        qv = test_imgs[qi]
        # cosine similarity against all train embeddings
        sims = [sum(a * b for a, b in zip(qv, emb)) for emb in embeddings]
        top_k_idxs = sorted(range(len(embeddings)), key=lambda i: sims[i], reverse=True)[:top_k]
        neighbor_labels = [embed_labels[i] for i in top_k_idxs]
        # majority vote
        votes = {}
        for lbl in neighbor_labels:
            votes[lbl] = votes.get(lbl, 0) + 1
        predicted = max(votes, key=votes.get)
        if predicted == test_lbls[qi]:
            correct += 1
        if len(examples) < 10:
            examples.append((test_lbls[qi], neighbor_labels))
        if (qi + 1) % 1000 == 0:
            log(f"  ... evaluated {qi + 1}/{total} test images")
    return correct, total, examples

# --- Checkpoint ---
# The previous version saved nothing, so a 22-hour run left no recoverable weights.
import json
def save_ckpt(path, sd):
    with open(path, 'w') as f:
        json.dump({k: v.data.data for k, v in sd.items()}, f)
save_ckpt('student_final.json', student_state_dict)
save_ckpt('teacher_final.json', teacher_state_dict)
log("\nSaved student_final.json / teacher_final.json")

# --- Evaluation ---
# Both probes run with train=False. Previously the post-head probe passed train=True, which applied the
# random RoPE box-jitter augmentation at inference and injected noise into the reported number.
# apply_head is passed explicitly so the head can be read without turning training behaviour back on.
def embed_all(imgs, sd, post_head):
    out = []
    for i, im in enumerate(imgs):
        o = vit(Raw(im), sd, train=False, apply_head=post_head)
        out.append(l2_norm((o[0] if post_head else o[2]).data).data[0])
        if (i + 1) % 1000 == 0:
            log(f"  ... embedded {i + 1}/{len(imgs)}")
    return out

def probe(name, sd, post_head, show_examples=True):
    db = embed_all(train_images[:KNN_IMAGES], sd, post_head)
    te = embed_all(test_images, sd, post_head)
    correct, total, examples = knn_evaluate(db, train_labels[:KNN_IMAGES], te, test_labels, TOP_K)
    log(f"{name}: {correct / total * 100:.1f}% ({correct}/{total})")
    if show_examples:
        for true_label, neighbor_labels in examples:
            log(f"  [{true_label}]: {neighbor_labels}")
    return correct / total * 100

log(f"\n--- KNN Evaluation ({KNN_IMAGES}-image database, top-{TOP_K}, {len(test_images)} test images) ---")
acc_pre  = probe(f"trained  pre-head CLS ({N_EMBED}-dim)", student_state_dict, post_head=False)
acc_post = probe(f"trained  post-head DINO ({HEAD_PROTOTYPES}-dim)", student_state_dict, post_head=True)

# --- Random-init control ---
# The honest baseline is the same architecture at initialization, not chance. init_state_dict is the
# exact starting point of this run, so the delta below is attributable to training and nothing else.
log("\n--- Random-init control (identical weights at step 0, zero training) ---")
acc_rand = probe(f"random   pre-head CLS ({N_EMBED}-dim)", init_state_dict, post_head=False, show_examples=False)

log(f"\nSummary: random-init {acc_rand:.1f}% | trained pre-head {acc_pre:.1f}% | trained post-head {acc_post:.1f}% | chance 10.0%")
log(f"Training delta over the random-init control: {acc_pre - acc_rand:+.1f} points")
