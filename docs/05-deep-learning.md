# Module 05 — Deep Learning & Neural Networks

> **Runnable code:** `src/05-deep-learning/`
> ```bash
> python src/05-deep-learning/nn_numpy.py
> python src/05-deep-learning/optimizers.py
> python src/05-deep-learning/mlflow_demo.py
> python src/05-deep-learning/monitoring.py
> ```

---

> **Python prerequisite:** This module uses Python, NumPy, and ML libraries throughout. If you need a foundation or refresher, visit the **Languages → Python** guide and read **Section 21 — Python for ML & AI** before starting.

## Prerequisites & Overview

**Prerequisites:** Modules 01–02 (matrix ops, partial derivatives, chain rule, gradient descent). NumPy only — all scripts run without PyTorch.
**Estimated time:** 10–15 hours

**Install:**
```bash
pip install numpy mlflow  # mlflow is optional
```

### Why This Module Matters

Every modern AI system — LLMs, diffusion models, vision transformers — is a deep neural network trained with backpropagation and adaptive optimizers. Understanding these from scratch means you can:
- Debug vanishing/exploding gradients
- Choose the right optimizer for each problem
- Explain gradient flow to interviewers
- Understand why certain architectures work

### Module Map

| Section | Core Concept | Why It Matters |
|---------|-------------|---------------|
| Neural network math | Forward pass, backprop | Debug NaNs, understand gradient flow |
| Activation functions | ReLU, GELU, sigmoid | Choose right activation per layer |
| Initialization | Xavier, He | Fix vanishing/exploding gradients |
| Optimizers | SGD → Adam → AdamW | Know when Adam fails |
| Regularization | Dropout, BatchNorm, LayerNorm | Control overfitting |
| MLflow | Experiment tracking | Reproduce any experiment by ID |

---

# 1. Neural Networks from Scratch

## Intuition

A neural network is a **chain of linear transformations with nonlinearities** between them.

Without nonlinearities, stacking layers would just be one big matrix multiplication — linear. Nonlinear activations (ReLU, sigmoid) let the network learn curved decision boundaries.

```
Input → [Linear → Activation] → [Linear → Activation] → [Linear] → Output
         Layer 1                  Layer 2                  Output layer
```

```
Detailed view (3 inputs → 4 hidden → 2 outputs):

Input layer    Hidden layer     Output layer
  x₁ ─┐          h₁ ─┐
       ├──[W¹]──▶      ├──[W²]──▶  ŷ₁
  x₂ ─┤          h₂ ─┤
       │          h₃ ─┤           ŷ₂
  x₃ ─┘          h₄ ─┘

Each arrow represents a weight (learnable parameter).
  z = W·x + b        (linear combination)
  a = ReLU(z)        (nonlinearity — introduces curves)

Why do we need multiple layers?
  1 layer  → can only learn lines (hyperplanes)
  2 layers → can approximate any continuous function (universal approximation theorem)
  More layers → can learn hierarchical features (edges → shapes → objects)
```

## 1.1 The Forward Pass

For layer $l$ with weight matrix $W^{(l)}$ and bias $\mathbf{b}^{(l)}$:

$$\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} \qquad \text{(pre-activation)}$$
$$\mathbf{a}^{(l)} = g^{(l)}(\mathbf{z}^{(l)}) \qquad \text{(post-activation)}$$

where $\mathbf{a}^{(0)} = \mathbf{x}$ (the input).

```python
import numpy as np

rng = np.random.default_rng(42)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip for numerical stability

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    # Subtract max for numerical stability (prevents exp overflow)
    z_shifted = z - z.max(axis=0, keepdims=True)
    exp_z     = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=0, keepdims=True)

# ── Build a 3-layer network ──────────────────────────────────
# Architecture: 4 inputs → 8 hidden → 4 hidden → 3 outputs (softmax)

np.random.seed(42)
W1 = np.random.randn(8, 4) * 0.01   # shape (n1, n0) = (8, 4)
b1 = np.zeros((8, 1))

W2 = np.random.randn(4, 8) * 0.01   # shape (n2, n1) = (4, 8)
b2 = np.zeros((4, 1))

W3 = np.random.randn(3, 4) * 0.01   # shape (n3, n2) = (3, 4)
b3 = np.zeros((3, 1))

def forward_pass(X):
    """
    X: shape (4, batch_size) — features as columns

    Returns all intermediate values (needed for backprop).
    """
    # Layer 1: Linear → ReLU
    z1 = W1 @ X + b1        # (8, batch_size)
    a1 = relu(z1)            # (8, batch_size)

    # Layer 2: Linear → ReLU
    z2 = W2 @ a1 + b2       # (4, batch_size)
    a2 = relu(z2)            # (4, batch_size)

    # Layer 3: Linear → Softmax (output)
    z3 = W3 @ a2 + b3       # (3, batch_size)
    a3 = softmax(z3)         # (3, batch_size) — probabilities over 3 classes

    return (z1, a1, z2, a2, z3, a3)

# Test with a batch of 5 samples
X_test = rng.randn(4, 5)   # 4 features, 5 samples
cache  = forward_pass(X_test)
a3     = cache[-1]          # output probabilities

print(f"Input shape:  {X_test.shape}")
print(f"Output shape: {a3.shape}")
print(f"\nPrediction probabilities (5 samples, 3 classes):")
print(a3.T.round(4))  # transpose for readability
print(f"\nEach row sums to: {a3.sum(axis=0).round(6)}")  # should be all 1.0
```

## 1.2 Activation Functions

```
Shapes of common activations:

Sigmoid σ(z)        ReLU max(0,z)       Tanh tanh(z)
    1 ┤···           ↑     /             1 ┤     ···
      │   ·          │    /                │    ·
  0.5 ┤─ · ─         │   /            0 ──┼── · ────▶
      │  ·           │  /                 │  ·
    0 ┤·             │ /              -1 ─┤···
      └──────        └──────              └──────
  range: (0,1)    range: [0,∞)        range: (-1,1)
  vanishes at ±5  no vanishing          zero-centered

GELU ≈ z·σ(1.702z)  ← smooth version of ReLU, used in GPT/BERT
```

```python
import numpy as np

# All activations + their derivatives
def sigmoid(z):
    s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    return s

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)          # max at z=0: 0.25

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)  # 1 if z>0, else 0

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_deriv(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

def tanh(z):
    return np.tanh(z)

def tanh_deriv(z):
    return 1 - np.tanh(z)**2

def gelu(z):
    """Gaussian Error Linear Unit — used in BERT, GPT."""
    return z * sigmoid(1.702 * z)   # approximate GELU

# Compare activation behaviors
z_vals = np.array([-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0])

print(f"{'z':>6} {'sigmoid':>10} {'ReLU':>8} {'tanh':>8} {'GELU':>8}")
print("-" * 44)
for z in z_vals:
    print(f"{z:>6.1f} {sigmoid(z):>10.4f} {relu(z):>8.4f} {tanh(z):>8.4f} {gelu(z):>8.4f}")

# Key insight: why ReLU > sigmoid for hidden layers
print("\nGradient at z=5 (far from boundary):")
print(f"  sigmoid'(5)  = {sigmoid_deriv(5):.6f}  ← nearly zero (vanishing gradient!)")
print(f"  relu'(5)     = {relu_deriv(5):.6f}  ← still 1 (gradient flows!)")
```

| Activation | Formula | Derivative | Dead neurons? | Use case |
|-----------|---------|-----------|--------------|---------|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $\sigma(1-\sigma)$ | No | Output (binary) |
| Tanh | $\tanh(z)$ | $1-\tanh^2(z)$ | No | RNN hidden |
| ReLU | $\max(0,z)$ | $\mathbf{1}[z>0]$ | Yes ($z<0$) | Default hidden |
| Leaky ReLU | $\max(0.01z, z)$ | $\in\{0.01, 1\}$ | Rarely | Sparse features |
| GELU | $z\Phi(z)$ | complex | No | Transformers |
| Softmax | $\frac{e^{z_k}}{\sum e^{z_j}}$ | Jacobian | N/A | Output (multiclass) |

---

# 2. Backpropagation — The Full Derivation

## Intuition

Backprop = **chain rule applied layer-by-layer from the output back to the input**. Each layer asks: "by how much does changing my input change the final loss?" That number is the gradient.

```
Forward pass (computing predictions):
  x ──▶ z¹=W¹x+b¹ ──▶ a¹=ReLU(z¹) ──▶ z²=W²a¹+b² ──▶ ŷ=σ(z²) ──▶ L

Backward pass (computing gradients via chain rule):
  ∂L/∂W¹ ◀── ∂L/∂z¹ ◀── ∂L/∂a¹ ◀── ∂L/∂z² ◀── ∂L/∂ŷ ◀── L
         ↑           ↑          ↑           ↑          ↑
     ×∂z¹/∂W¹  ×∂a¹/∂z¹  ×∂z²/∂a¹  ×∂ŷ/∂z²  ×∂L/∂ŷ
     = a⁰ᵀ      = ReLU'    = W²ᵀ       = ŷ(1-ŷ)  = ŷ-y

Key insight: gradients flow backward through the same graph as the forward pass.
             Each node multiplies its local gradient with the incoming upstream gradient.
```

## 2.1 Define the Error Signals

For each layer $l$, define the upstream gradient:
$$\boldsymbol{\delta}^{(l)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}$$

**Output layer** (softmax + categorical cross-entropy — "the miracle"):
$$\boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$$

The Jacobian of softmax exactly cancels the derivative of cross-entropy, leaving prediction minus label.

**Hidden layer** (chain rule backwards through activation $g$):
$$\boldsymbol{\delta}^{(l)} = \left(W^{(l+1)T} \boldsymbol{\delta}^{(l+1)}\right) \odot g'^{(l)}(\mathbf{z}^{(l)})$$

**Parameter gradients:**
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{1}{m} \boldsymbol{\delta}^{(l)} \mathbf{a}^{(l-1)T} \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \frac{1}{m} \boldsymbol{\delta}^{(l)} \mathbf{1}$$

## 2.2 Full Neural Network from Scratch

```python
import numpy as np

class NeuralNetwork:
    """2-layer NN trained by backprop. Pure NumPy, no frameworks."""

    def __init__(self, layer_sizes, lr=0.01):
        """
        layer_sizes: e.g. [input, hidden, output] = [4, 8, 3]
        """
        self.lr     = lr
        self.params = {}
        self.L      = len(layer_sizes) - 1  # number of weight layers

        rng = np.random.default_rng(42)

        # Xavier initialization: scale by sqrt(2 / (n_in + n_out))
        for l in range(1, len(layer_sizes)):
            n_in  = layer_sizes[l-1]
            n_out = layer_sizes[l]
            scale = np.sqrt(2.0 / (n_in + n_out))

            self.params[f'W{l}'] = rng.normal(0, scale, (n_out, n_in))
            self.params[f'b{l}'] = np.zeros((n_out, 1))

    def forward(self, X):
        """
        X: shape (n_features, batch_size)
        Returns predictions and cache of intermediates for backprop.
        """
        cache = {'a0': X}
        a     = X

        for l in range(1, self.L + 1):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']

            z = W @ a + b           # pre-activation
            cache[f'z{l}'] = z

            # Hidden layers: ReLU. Output layer: softmax.
            if l < self.L:
                a = np.maximum(0, z)   # ReLU
            else:
                # Stable softmax
                z_shifted = z - z.max(axis=0, keepdims=True)
                exp_z     = np.exp(z_shifted)
                a         = exp_z / exp_z.sum(axis=0, keepdims=True)

            cache[f'a{l}'] = a

        return a, cache   # a = final predictions

    def compute_loss(self, y_hat, y_true):
        """
        y_hat: predictions, shape (n_classes, batch_size)
        y_true: one-hot labels, shape (n_classes, batch_size)
        """
        m   = y_true.shape[1]
        eps = 1e-9
        ce  = -np.sum(y_true * np.log(y_hat + eps)) / m
        return ce

    def backward(self, y_true, cache):
        """Returns gradients for all parameters."""
        grads = {}
        m     = y_true.shape[1]

        # Output layer: δ = ŷ - y  (softmax + cross-entropy miracle)
        delta = cache[f'a{self.L}'] - y_true   # shape (n_out, m)

        for l in range(self.L, 0, -1):
            a_prev = cache[f'a{l-1}']

            # Gradients for W and b at layer l
            grads[f'W{l}'] = (1/m) * delta @ a_prev.T
            grads[f'b{l}'] = (1/m) * delta.sum(axis=1, keepdims=True)

            if l > 1:
                # Propagate error backwards: δ_{l-1} = W_l^T δ_l ⊙ ReLU'(z_{l-1})
                W      = self.params[f'W{l}']
                z_prev = cache[f'z{l-1}']
                delta  = (W.T @ delta) * (z_prev > 0)  # ReLU derivative

        return grads

    def update_params(self, grads):
        """Standard gradient descent update."""
        for l in range(1, self.L + 1):
            self.params[f'W{l}'] -= self.lr * grads[f'W{l}']
            self.params[f'b{l}'] -= self.lr * grads[f'b{l}']

    def fit(self, X, y_one_hot, epochs=500, batch_size=32, verbose=True):
        """
        X: shape (n_features, n_samples)
        y_one_hot: shape (n_classes, n_samples)
        """
        n       = X.shape[1]
        history = []

        for epoch in range(epochs):
            # Shuffle
            idx    = np.random.permutation(n)
            X_s    = X[:, idx]
            y_s    = y_one_hot[:, idx]

            epoch_loss = 0
            for start in range(0, n, batch_size):
                X_b  = X_s[:, start:start+batch_size]
                y_b  = y_s[:, start:start+batch_size]

                y_hat, cache = self.forward(X_b)
                loss         = self.compute_loss(y_hat, y_b)
                grads        = self.backward(y_b, cache)
                self.update_params(grads)

                epoch_loss += loss

            avg_loss = epoch_loss / (n // batch_size)
            history.append(avg_loss)

            if verbose and epoch % 100 == 0:
                y_pred = self.predict(X)
                acc    = (y_pred == y_one_hot.argmax(axis=0)).mean()
                print(f"Epoch {epoch:4d}: loss={avg_loss:.4f}, acc={acc:.3f}")

        return history

    def predict(self, X):
        """Returns class indices."""
        y_hat, _ = self.forward(X)
        return y_hat.argmax(axis=0)

# ── Train on Iris ──────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X    = iris.data.T           # shape (4, 150) — features as rows
y    = iris.target           # shape (150,)

# One-hot encode labels
def one_hot(y, n_classes):
    oh = np.zeros((n_classes, len(y)))
    oh[y, np.arange(len(y))] = 1
    return oh

y_oh = one_hot(y, 3)        # shape (3, 150)

# Normalize
scaler = StandardScaler()
X_norm = scaler.fit_transform(X.T).T  # normalize features, keep shape

# Train
net     = NeuralNetwork([4, 16, 8, 3], lr=0.05)
history = net.fit(X_norm, y_oh, epochs=300, batch_size=32)

# Evaluate
y_pred = net.predict(X_norm)
acc    = (y_pred == y).mean()
print(f"\nFinal accuracy: {acc:.3f}")
```

---

# 3. Weight Initialization

## Why It Matters

**Zero initialization:** All neurons compute identical gradients → equivalent to a single neuron per layer. Symmetry never breaks.

**Random too small:** Activations shrink layer by layer → **vanishing gradients** (no learning in early layers).

**Random too large:** Activations explode layer by layer → **exploding gradients** (NaNs).

```python
import numpy as np

rng = np.random.default_rng(42)

def simulate_forward(W_init_fn, n_layers=10, n=100):
    """Show how activations grow/shrink through layers."""
    a = rng.randn(n, 1)  # input
    stds = [a.std()]

    for _ in range(n_layers):
        W = W_init_fn(n, n)
        a = np.tanh(W @ a)  # tanh hidden layers
        stds.append(a.std())

    return stds

# Zero init
zero_stds = simulate_forward(lambda n, m: np.zeros((n, m)))

# Too small
small_stds = simulate_forward(lambda n, m: rng.randn(n, m) * 0.01)

# Too large
large_stds = simulate_forward(lambda n, m: rng.randn(n, m) * 1.0)

# Xavier: sqrt(1/n_in) for tanh
xavier_stds = simulate_forward(lambda n, m: rng.randn(n, m) * np.sqrt(1.0/m))

print("Activation std across 10 layers:")
print(f"{'Layer':<8} {'Zero':>8} {'Small':>8} {'Large':>12} {'Xavier':>10}")
print("-" * 50)
for l, (z, s, lg, x) in enumerate(zip(zero_stds, small_stds, large_stds, xavier_stds)):
    print(f"{l:<8} {z:>8.4f} {s:>8.4f} {lg:>12.4f} {x:>10.4f}")
```

| Init | Formula | Good for |
|------|---------|---------|
| Xavier/Glorot | $\mathcal{N}(0, \sqrt{2/(n_{in}+n_{out})})$ | Sigmoid/Tanh |
| He/Kaiming | $\mathcal{N}(0, \sqrt{2/n_{in}})$ | ReLU, Leaky ReLU |
| Lecun | $\mathcal{N}(0, \sqrt{1/n_{in}})$ | SELU |

```python
def xavier_init(n_in, n_out):
    scale = np.sqrt(2.0 / (n_in + n_out))
    return rng.normal(0, scale, (n_out, n_in))

def he_init(n_in, n_out):
    scale = np.sqrt(2.0 / n_in)   # 2 because ReLU kills half the neurons
    return rng.normal(0, scale, (n_out, n_in))

# Verify: std of pre-activations should be ~1 across layers with He init
W = he_init(128, 256)
x = rng.randn(128, 1)
z = W @ x
print(f"\nHe init: z.std() = {z.std():.4f} (target ~1.0)")
```

---

# 4. Optimizers

## Intuition

Gradient descent with fixed learning rate is slow and sensitive. Adaptive optimizers:
- Use **momentum** to accelerate in consistent directions
- Adapt **per-parameter learning rates** based on gradient history
- Handle sparse gradients better (useful for embeddings)

```
Optimizer behavior on a loss landscape:

SGD (vanilla):              SGD + Momentum:         Adam:
↓ → ↓ → ↓ → ↓              ↘ ↘ ↘ ↘ ↘ ↘            → → → →
  zig-zag                    builds speed            smooth,
  slowly                     like a ball             adaptive
                             rolling downhill        per-param lr

Loss ↑                      Loss ↑                 Loss ↑
     │ ↓ ↓ ↓                     │ ↘↘↘↘                │ ────
     └──────▶ steps               └──────▶ steps         └──────▶ steps
    many oscillations           fewer oscillations    fewest steps

Rule of thumb: start with Adam, switch to SGD+momentum for fine-tuning.
```

## 4.1 Optimizer Implementations from Scratch

```python
import numpy as np

class SGD:
    """Vanilla stochastic gradient descent."""
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]
        return params

class SGDMomentum:
    """SGD with momentum — accumulates velocity in gradient direction."""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr       = lr
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])

            # v = β*v + (1-β)*g  (or sometimes β*v + g — both work)
            self.velocity[key] = self.momentum * self.velocity[key] + grads[key]
            params[key] -= self.lr * self.velocity[key]

        return params

class Adam:
    """
    Adam: Adaptive Moment Estimation.
    Tracks first moment (mean of gradient) and second moment (mean of gradient²).
    Bias correction handles the fact that m and v start at zero.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1    # first moment decay (momentum)
        self.beta2 = beta2    # second moment decay (RMSProp)
        self.eps   = eps      # prevents division by zero
        self.m     = {}       # first moment estimate
        self.v     = {}       # second moment estimate
        self.t     = 0        # timestep

    def update(self, params, grads):
        self.t += 1

        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            g = grads[key]

            # Update biased moment estimates
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g**2

            # Bias correction (compensate for zero initialization)
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # Parameter update
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return params

class AdamW:
    """Adam with decoupled weight decay — standard in Transformers."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.m, self.v    = {}, {}
        self.t            = 0

    def update(self, params, grads):
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            g = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g**2

            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # AdamW: weight decay applied BEFORE gradient, not inside it
            params[key] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps)
                                      + self.weight_decay * params[key])

        return params

# ── Compare optimizers on a simple loss landscape ─────────────
def rosenbrock(x, y):
    """Rosenbrock function — notoriously hard to optimize. Minimum at (1, 1)."""
    return (1 - x)**2 + 100*(y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2*(1-x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

def optimize_rosenbrock(optimizer_class, **kwargs):
    params = {'xy': np.array([-1.0, 2.0])}   # starting point
    opt    = optimizer_class(**kwargs)
    losses = []

    for step in range(5000):
        x, y = params['xy']
        loss = rosenbrock(x, y)
        losses.append(loss)

        grad = rosenbrock_grad(x, y)
        params = opt.update(params, {'xy': grad})

    final_x, final_y = params['xy']
    return losses, final_x, final_y

print("Optimizer comparison on Rosenbrock function (min at (1,1)):")
print(f"{'Optimizer':<15} {'Final Loss':>12} {'x':>8} {'y':>8}")
print("-" * 45)

configs = [
    (SGD,          "SGD",         {'lr': 0.001}),
    (SGDMomentum,  "Momentum",    {'lr': 0.001, 'momentum': 0.9}),
    (Adam,         "Adam",        {'lr': 0.01}),
    (AdamW,        "AdamW",       {'lr': 0.01, 'weight_decay': 0.001}),
]

for OptimizerClass, name, kwargs in configs:
    losses, fx, fy = optimize_rosenbrock(OptimizerClass, **kwargs)
    print(f"{name:<15} {losses[-1]:>12.6f} {fx:>8.4f} {fy:>8.4f}")
```

## 4.2 Learning Rate Schedules

```python
import numpy as np

def warmup_cosine_schedule(step, warmup_steps, total_steps, max_lr=1e-3, min_lr=1e-6):
    """Warmup + cosine decay — standard in Transformer training."""
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))

def step_schedule(step, milestones, gamma=0.1, initial_lr=0.01):
    """Reduce LR by gamma at each milestone."""
    lr = initial_lr
    for m in milestones:
        if step >= m:
            lr *= gamma
    return lr

# Visualize schedules
steps = np.arange(0, 1000)
warmup_lr = [warmup_cosine_schedule(s, warmup_steps=100, total_steps=1000) for s in steps]
step_lr   = [step_schedule(s, milestones=[400, 700]) for s in steps]

print("Learning rate schedule values:")
print(f"{'Step':>8} {'Warmup+Cosine':>15} {'Step Decay':>12}")
for s in [0, 50, 100, 200, 400, 600, 700, 900, 999]:
    print(f"{s:>8} {warmup_lr[s]:>15.6f} {step_lr[s]:>12.6f}")
```

---

# 5. Regularization

## 5.1 Dropout

```python
import numpy as np

def dropout(a, keep_prob=0.8, training=True):
    """
    Randomly zero out neurons during training.
    At inference: multiply by keep_prob (or use inverted dropout).
    """
    if not training:
        return a  # no dropout at inference

    # Inverted dropout: scale up during training → no need to scale at test
    mask = (np.random.rand(*a.shape) < keep_prob) / keep_prob
    return a * mask

# Demo
rng = np.random.default_rng(42)
activations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

print("Dropout demo (keep_prob=0.6):")
for trial in range(5):
    dropped = dropout(activations, keep_prob=0.6)
    print(f"  Trial {trial+1}: {dropped.round(3)}")

print(f"\nInference (no dropout): {dropout(activations, training=False)}")
```

## 5.2 Batch Normalization

**Intuition:** Normalize activations within a mini-batch so each layer sees stable, zero-mean, unit-variance inputs. Reduces sensitivity to initialization and learning rate.

$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \qquad y_i = \gamma \hat{z}_i + \beta$$

```python
import numpy as np

class BatchNorm:
    """Batch normalization layer."""

    def __init__(self, n_features, eps=1e-8, momentum=0.1):
        self.eps      = eps
        self.momentum = momentum
        self.gamma    = np.ones(n_features)    # scale
        self.beta     = np.zeros(n_features)   # shift

        # Running stats for inference
        self.running_mean = np.zeros(n_features)
        self.running_var  = np.ones(n_features)

        self._cache = None

    def forward(self, X, training=True):
        """
        X: shape (batch_size, n_features)
        """
        if training:
            mu    = X.mean(axis=0)
            var   = X.var(axis=0)

            # Update running stats for inference
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mu  = self.running_mean
            var = self.running_var

        # Normalize
        X_norm = (X - mu) / np.sqrt(var + self.eps)

        # Scale and shift (learnable)
        out = self.gamma * X_norm + self.beta

        self._cache = (X, X_norm, mu, var)
        return out

# Demo
rng  = np.random.default_rng(42)
bn   = BatchNorm(4)

# Before BN: wildly different scales
X = np.array([
    [100, 0.01, 5000, 0.1],
    [120, 0.02, 4800, 0.2],
    [80,  0.01, 5200, 0.15],
    [110, 0.03, 4900, 0.05],
])

X_out = bn.forward(X)
print("Before BatchNorm:")
print(f"  Mean per feature:  {X.mean(axis=0).round(2)}")
print(f"  Std per feature:   {X.std(axis=0).round(2)}")
print(f"\nAfter BatchNorm:")
print(f"  Mean per feature:  {X_out.mean(axis=0).round(6)}")  # ~0
print(f"  Std per feature:   {X_out.std(axis=0).round(6)}")   # ~1
```

**BatchNorm vs LayerNorm:**

| | BatchNorm | LayerNorm |
|-|-----------|-----------|
| Normalize over | Mini-batch (across samples) | Features (within one sample) |
| Best for | CNNs, fully-connected | Transformers, RNNs |
| Issues | Needs batch size > 1; doesn't work with online learning | None |
| Inference | Uses running mean/var | Same as training |

---

# 6. MLflow — Experiment Tracking

## Intuition

ML experiments multiply fast. Without tracking, you don't know which hyperparameters produced your best model. MLflow logs every experiment so you can:
- Compare runs
- Reproduce any result
- Register the best model

```python
try:
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np

    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # MLflow experiment
    mlflow.set_experiment("iris-classification")

    experiments = [
        {"C": 0.1,  "solver": "lbfgs"},
        {"C": 1.0,  "solver": "lbfgs"},
        {"C": 10.0, "solver": "lbfgs"},
    ]

    for params in experiments:
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("dataset", "iris")

            # Train
            model = LogisticRegression(**params, max_iter=1000)
            model.fit(X_tr_s, y_tr)

            # Evaluate
            y_pred = model.predict(X_te_s)
            acc    = accuracy_score(y_te, y_pred)
            f1     = f1_score(y_te, y_pred, average='macro')

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            print(f"C={params['C']}: accuracy={acc:.4f}, f1={f1:.4f}")

    print("\nAll runs logged! View at: mlflow ui (then open localhost:5000)")

except ImportError:
    print("MLflow not installed. Run: pip install mlflow")
    print("MLflow usage shown conceptually above.")
```

---

# 7. Interview Q&A

## Q1: Explain vanishing gradients and how to fix them.

In deep networks with sigmoid/tanh, the derivative is at most 0.25. Multiplied across 10 layers via chain rule: $0.25^{10} \approx 10^{-6}$ — early layers receive almost no gradient signal. Fixes: (1) ReLU activations (gradient is 1 for positive inputs), (2) Residual connections (gradient bypasses nonlinearities), (3) Proper initialization (Xavier/He), (4) BatchNorm (stabilizes activation distributions).

## Q2: Why does Adam converge faster than SGD?

Adam maintains per-parameter adaptive learning rates. Parameters with consistently large gradients get small effective LR; sparse parameters get large LR. The momentum term smooths oscillations. In practice, Adam requires less LR tuning and converges in far fewer steps. However, Adam can generalize worse than SGD on vision tasks — hence the common pattern: Adam for fast convergence early, SGD for final fine-tuning.

## Q3: What is the difference between BatchNorm and LayerNorm?

BatchNorm normalizes across the **batch dimension** (statistics computed over all samples for each feature). LayerNorm normalizes across the **feature dimension** (statistics computed over all features for each sample). BatchNorm is effective for CNNs but breaks with batch size 1 and doesn't work well for variable-length sequences. LayerNorm is batch-size independent, making it the standard for Transformers and RNNs.

## Q4: Why is the output layer gradient for softmax + cross-entropy so simple?

The cross-entropy gradient through softmax simplifies to $\hat{y}_k - y_k$. This happens because the Jacobian of softmax has a specific structure that cancels with the derivative of the log in cross-entropy. The "miracle" result makes backprop implementation clean and numerically stable.

## Q5: When should you use dropout vs L2 regularization?

**L2:** Smooth penalty, always active, doesn't change network architecture. Best for simple models.
**Dropout:** Forces redundant representations, acts as ensemble of $2^n$ sub-networks. Best for large overparameterized models. Not effective on small datasets. Never use dropout on BatchNorm layers.

---

# 8. Cheat Sheet

| Concept | Formula | ML Role |
|---------|---------|---------|
| Forward pass | $\mathbf{a}^{(l)} = g(W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)})$ | Compute predictions |
| Output gradient | $\boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$ | Start backprop |
| Hidden gradient | $\boldsymbol{\delta}^{(l)} = W^{(l+1)T}\boldsymbol{\delta}^{(l+1)} \odot g'(\mathbf{z}^{(l)})$ | Chain rule |
| Weight gradient | $\frac{1}{m}\boldsymbol{\delta}^{(l)}\mathbf{a}^{(l-1)T}$ | Update weights |
| Xavier init | $\sigma = \sqrt{2/(n_{in}+n_{out})}$ | Sigmoid/Tanh layers |
| He init | $\sigma = \sqrt{2/n_{in}}$ | ReLU layers |
| Adam update | $w \mathrel{-}= \alpha \hat{m}/(\sqrt{\hat{v}} + \epsilon)$ | Adaptive learning |
| BatchNorm | $\hat{z} = (z - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$ | Stable training |
| Dropout | Zero each neuron with prob $1-p$, scale by $1/p$ | Regularization |

---

# MINI-PROJECT — MNIST Digit Classifier from Scratch

**What you will build:** A 3-layer neural network that classifies handwritten digits (0–9) trained entirely with backpropagation on NumPy. No PyTorch, no Keras.

---

## Step 1 — Generate Synthetic MNIST-Like Data

```python
import numpy as np
from sklearn.datasets import load_digits  # 8x8 version of MNIST, built into sklearn

digits   = load_digits()
X_raw    = digits.data          # shape (1797, 64) — 8x8 images flattened
y_raw    = digits.target        # shape (1797,)  — labels 0-9

# Normalize pixel values to [0, 1]
X = X_raw / 16.0               # max pixel value is 16 in sklearn digits

print(f"Dataset: {X.shape[0]} images, {X.shape[1]} features ({int(X.shape[1]**0.5)}x{int(X.shape[1]**0.5)} pixels)")
print(f"Classes: {np.unique(y_raw)} ({len(np.unique(y_raw))} digits)")

# One-hot encode
def one_hot(y, n_classes=10):
    oh = np.zeros((n_classes, len(y)))
    oh[y, np.arange(len(y))] = 1
    return oh

# Train/test split (80/20)
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
split    = int(0.8 * len(X))

X_train  = X[idx[:split]].T      # shape (64, n_train) — transpose for our NN
y_train  = y_raw[idx[:split]]
y_tr_oh  = one_hot(y_train)      # shape (10, n_train)

X_test   = X[idx[split:]].T      # shape (64, n_test)
y_test   = y_raw[idx[split:]]

print(f"\nTrain: {X_train.shape[1]} samples")
print(f"Test:  {X_test.shape[1]} samples")
```

---

## Step 2 — Initialize and Train

```python
# Use the NeuralNetwork class from Section 2

net = NeuralNetwork(
    layer_sizes=[64, 128, 64, 10],   # 64 inputs, 2 hidden layers, 10 outputs
    lr=0.05
)

print("Training digit classifier...")
history = net.fit(X_train, y_tr_oh, epochs=500, batch_size=64, verbose=True)
```

---

## Step 3 — Evaluate

```python
# Test set accuracy
y_pred_test = net.predict(X_test)
test_acc    = (y_pred_test == y_test).mean()
print(f"\nTest accuracy: {test_acc:.4f}")

# Confusion-matrix style: per-class accuracy
print("\nPer-digit accuracy:")
for digit in range(10):
    mask      = y_test == digit
    digit_acc = (y_pred_test[mask] == digit).mean()
    bar       = "█" * int(digit_acc * 20)
    print(f"  Digit {digit}: {digit_acc:.2f}  {bar}")

# Loss curve summary
print(f"\nLoss: {history[0]:.4f} → {history[-1]:.4f}")
print(f"Improvement: {(1 - history[-1]/history[0])*100:.1f}%")
```

---

## Step 4 — Visualize Some Predictions

```python
import numpy as np

# Print digit images as ASCII art
def show_digit(pixel_row, label, predicted):
    pixels = pixel_row * 16  # scale back
    print(f"  True: {label}, Predicted: {predicted} {'✓' if label == predicted else '✗'}")
    for row in range(8):
        line = ""
        for col in range(8):
            val = pixels[row*8 + col]
            line += "█" if val > 8 else "░" if val > 2 else " "
        print(f"  {line}")
    print()

# Show 5 test samples
print("Sample predictions:")
sample_idx = rng.choice(X_test.shape[1], 5, replace=False)

y_hat_probs = net.forward(X_test)[0]  # probabilities
for i in sample_idx:
    pixel_row = X_test[:, i]
    true_lbl  = y_test[i]
    pred_lbl  = y_pred_test[i]
    confidence = y_hat_probs[pred_lbl, i]

    show_digit(pixel_row, true_lbl, pred_lbl)
    print(f"  Confidence: {confidence:.3f}")
```

---

## Step 5 — Experiment with Hyperparameters

```python
configs = [
    {"layer_sizes": [64, 32, 10],         "lr": 0.05, "name": "Tiny"},
    {"layer_sizes": [64, 128, 64, 10],    "lr": 0.05, "name": "Medium"},
    {"layer_sizes": [64, 256, 128, 10],   "lr": 0.01, "name": "Large"},
    {"layer_sizes": [64, 128, 64, 10],    "lr": 0.1,  "name": "Med+HighLR"},
]

print("\nHyperparameter experiment:")
print(f"{'Config':<15} {'Test Acc':>10} {'Final Loss':>12}")
print("-" * 40)

for cfg in configs:
    name  = cfg.pop("name")
    model = NeuralNetwork(**cfg)
    hist  = model.fit(X_train, y_tr_oh, epochs=200, batch_size=64, verbose=False)
    acc   = (model.predict(X_test) == y_test).mean()
    print(f"{name:<15} {acc:>10.4f} {hist[-1]:>12.4f}")
    cfg["name"] = name  # restore
```

---

## What This Project Demonstrated

| Concept | Where it appeared |
|---------|------------------|
| Forward pass | `net.forward(X_train)` — layers × activations |
| Backpropagation | `net.backward(y_oh, cache)` — chain rule all the way |
| Xavier init | `NeuralNetwork.__init__` — `scale = sqrt(2/(n_in+n_out))` |
| Softmax output | Final layer of forward pass |
| Categorical cross-entropy | `compute_loss` |
| Output gradient shortcut | `delta_L = y_hat - y_true` |
| Mini-batch gradient descent | 64-sample batches per update |
| Hyperparameter search | Tested 4 configurations |
| One-hot encoding | `one_hot(y, 10)` |

You built a digit classifier from mathematical primitives — the same operations (with more engineering) power every neural network in production.

---

*Next: [Module 06 — GenAI Core](06-genai-core.md)*
