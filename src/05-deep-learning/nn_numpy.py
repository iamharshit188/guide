"""
2-layer neural network — forward pass, backpropagation, full training loop, pure NumPy.
pip install numpy
"""

import numpy as np


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Activation functions ───────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(np.float64)

def sigmoid(z):
    # Numerically stable: avoid overflow for large negative z
    pos = z >= 0
    out = np.zeros_like(z, dtype=np.float64)
    out[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_neg = np.exp(z[~pos])
    out[~pos] = exp_neg / (1.0 + exp_neg)
    return out

def sigmoid_grad(a):
    return a * (1 - a)

def softmax(z):
    # Subtract row-max for numerical stability
    z_shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(z_shifted)
    return e / e.sum(axis=1, keepdims=True)

def tanh_fn(z):
    return np.tanh(z)

def tanh_grad(a):
    return 1 - a**2


# ── Loss functions ─────────────────────────────────────────────

def binary_cross_entropy(y_hat, y):
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def categorical_cross_entropy(y_hat, y_one_hot):
    eps = 1e-15
    return -np.mean(np.sum(y_one_hot * np.log(np.clip(y_hat, eps, 1)), axis=1))


# ── Weight initialisation ──────────────────────────────────────

def he_init(fan_in, fan_out, rng):
    """He (Kaiming) initialisation for ReLU layers: N(0, 2/fan_in)."""
    return rng.standard_normal((fan_out, fan_in)) * np.sqrt(2.0 / fan_in)

def xavier_init(fan_in, fan_out, rng):
    """Xavier/Glorot initialisation for sigmoid/tanh: N(0, 2/(fan_in+fan_out))."""
    return rng.standard_normal((fan_out, fan_in)) * np.sqrt(2.0 / (fan_in + fan_out))


# ── 2-Layer NN (one hidden layer) — explicit, educational ─────

class TwoLayerNN:
    """
    Architecture: input → [W1, b1] → ReLU → [W2, b2] → Sigmoid → output

    For binary classification.
    Stores all forward-pass intermediates needed for backprop.
    """

    def __init__(self, n_input, n_hidden, n_output=1, seed=42):
        rng = np.random.default_rng(seed)
        # He init for hidden (ReLU), Xavier for output (sigmoid)
        self.W1 = he_init(n_input,  n_hidden, rng)   # (n_hidden, n_input)
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = xavier_init(n_hidden, n_output, rng)  # (n_output, n_hidden)
        self.b2 = np.zeros((n_output, 1))
        # Cache for backprop
        self._cache = {}

    def forward(self, X):
        """
        X: (n_input, m)  — m examples, n_input features per column.
        Returns y_hat: (1, m)
        """
        # Layer 1
        Z1 = self.W1 @ X + self.b1    # (n_hidden, m)
        A1 = relu(Z1)                  # (n_hidden, m)
        # Layer 2
        Z2 = self.W2 @ A1 + self.b2   # (1, m)
        A2 = sigmoid(Z2)               # (1, m)

        self._cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2

    def compute_loss(self, Y):
        """Binary cross-entropy. Y: (1, m)."""
        A2 = self._cache["A2"]
        return binary_cross_entropy(A2, Y)

    def backward(self, Y):
        """
        Returns dict of gradients: dW1, db1, dW2, db2.
        Y: (1, m)
        """
        m = Y.shape[1]
        X, Z1, A1, Z2, A2 = (self._cache[k] for k in ("X","Z1","A1","Z2","A2"))

        # --- Layer 2 ---
        # dL/dZ2 = A2 - Y   (BCE + sigmoid: Jacobian cancels)
        dZ2 = A2 - Y                          # (1, m)
        dW2 = (dZ2 @ A1.T) / m               # (1, n_hidden)
        db2 = dZ2.mean(axis=1, keepdims=True) # (1, 1)

        # --- Layer 1 ---
        # dL/dA1 = W2^T @ dZ2
        dA1 = self.W2.T @ dZ2                 # (n_hidden, m)
        # dL/dZ1 = dL/dA1 * ReLU'(Z1)
        dZ1 = dA1 * relu_grad(Z1)             # (n_hidden, m)
        dW1 = (dZ1 @ X.T) / m                 # (n_hidden, n_input)
        db1 = dZ1.mean(axis=1, keepdims=True) # (n_hidden, 1)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update(self, grads, lr):
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int)

    def accuracy(self, X, Y):
        preds = self.predict(X)
        return np.mean(preds == Y)


# ── L-layer generalised network ────────────────────────────────

class DeepNN:
    """
    Arbitrary depth network: [ReLU hidden layers] + [Sigmoid output].
    layer_dims: list of ints, e.g. [n_input, 64, 32, 1].
    """

    def __init__(self, layer_dims, seed=42):
        rng = np.random.default_rng(seed)
        L = len(layer_dims) - 1
        self.params = {}
        for l in range(1, L + 1):
            fan_in  = layer_dims[l-1]
            fan_out = layer_dims[l]
            if l < L:
                self.params[f"W{l}"] = he_init(fan_in, fan_out, rng)
            else:
                self.params[f"W{l}"] = xavier_init(fan_in, fan_out, rng)
            self.params[f"b{l}"] = np.zeros((fan_out, 1))
        self.L = L
        self._cache = {}

    def forward(self, X):
        A = X
        caches = {}
        for l in range(1, self.L):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z = W @ A + b
            A_prev = A
            A = relu(Z)
            caches[l] = {"A_prev": A_prev, "Z": Z}

        # Output layer
        Wl = self.params[f"W{self.L}"]
        bl = self.params[f"b{self.L}"]
        Zl = Wl @ A + bl
        Al = sigmoid(Zl)
        caches[self.L] = {"A_prev": A, "Z": Zl}
        self._cache = caches
        return Al

    def compute_loss(self, Y):
        Al = self._cache[self.L]["Z"]  # we stored Z not A for last layer
        # get sigmoid output
        eps = 1e-15
        A_hat = sigmoid(Al)
        A_hat = np.clip(A_hat, eps, 1 - eps)
        return -np.mean(Y * np.log(A_hat) + (1 - Y) * np.log(1 - A_hat))

    def backward(self, Y):
        m = Y.shape[1]
        grads = {}

        # Output layer
        Zl = self._cache[self.L]["Z"]
        Al = sigmoid(Zl)
        A_prev_L = self._cache[self.L]["A_prev"]
        dZl = Al - Y
        grads[f"dW{self.L}"] = (dZl @ A_prev_L.T) / m
        grads[f"db{self.L}"] = dZl.mean(axis=1, keepdims=True)
        dA = self.params[f"W{self.L}"].T @ dZl

        # Hidden layers in reverse
        for l in range(self.L - 1, 0, -1):
            Z = self._cache[l]["Z"]
            A_prev = self._cache[l]["A_prev"]
            dZ = dA * relu_grad(Z)
            grads[f"dW{l}"] = (dZ @ A_prev.T) / m
            grads[f"db{l}"] = dZ.mean(axis=1, keepdims=True)
            dA = self.params[f"W{l}"].T @ dZ

        return grads

    def update(self, grads, lr):
        for l in range(1, self.L + 1):
            self.params[f"W{l}"] -= lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= lr * grads[f"db{l}"]

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int)

    def accuracy(self, X, Y):
        return np.mean(self.predict(X) == Y)


# ── Numerical gradient check ───────────────────────────────────

def numerical_gradient_check(model, X, Y, eps=1e-5):
    """
    Compare analytical gradients from backprop against
    central-difference numerical gradients.
    Relative error < 1e-5 confirms correct implementation.
    """
    model.forward(X)
    analytical = model.backward(Y)

    max_errors = {}
    for key in [k for k in analytical if "W" in k]:
        param_key = key[1:]  # "dW1" → "W1"
        if isinstance(model, TwoLayerNN):
            param = getattr(model, param_key)
        else:
            param = model.params[param_key]

        num_grad = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        count = 0
        while not it.finished and count < 50:  # sample 50 elements
            idx = it.multi_index
            orig = param[idx]

            param[idx] = orig + eps
            model.forward(X)
            loss_plus = model.compute_loss(Y)

            param[idx] = orig - eps
            model.forward(X)
            loss_minus = model.compute_loss(Y)

            num_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            param[idx] = orig
            it.iternext()
            count += 1

        ana = analytical[key]
        # relative error over the sampled elements
        diff = np.abs(ana - num_grad)
        denom = np.abs(ana) + np.abs(num_grad) + 1e-20
        rel_err = (diff / denom).max()
        max_errors[key] = rel_err

    # Restore correct forward pass
    model.forward(X)
    return max_errors


# ── Training loop ──────────────────────────────────────────────

def train(model, X_train, Y_train, X_val, Y_val,
          lr=0.01, n_epochs=500, batch_size=64, print_every=100):
    m = X_train.shape[1]
    rng = np.random.default_rng(0)
    history = []

    for epoch in range(1, n_epochs + 1):
        # Mini-batch SGD
        idx = rng.permutation(m)
        X_sh, Y_sh = X_train[:, idx], Y_train[:, idx]
        batch_losses = []

        for start in range(0, m, batch_size):
            Xb = X_sh[:, start:start+batch_size]
            Yb = Y_sh[:, start:start+batch_size]
            model.forward(Xb)
            loss = model.compute_loss(Yb)
            grads = model.backward(Yb)
            model.update(grads, lr)
            batch_losses.append(loss)

        train_loss = np.mean(batch_losses)
        val_acc    = model.accuracy(X_val, Y_val)
        train_acc  = model.accuracy(X_train, Y_train)
        history.append({"epoch": epoch, "loss": train_loss,
                         "train_acc": train_acc, "val_acc": val_acc})

        if epoch % print_every == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    return history


# ── Synthetic dataset ──────────────────────────────────────────

def make_circles(n=1000, noise=0.05, seed=42):
    """Two concentric circles — non-linearly separable binary classification."""
    rng = np.random.default_rng(seed)
    n_half = n // 2
    theta = rng.uniform(0, 2 * np.pi, n_half)
    r_inner = 0.5 + rng.normal(0, noise, n_half)
    r_outer = 1.0 + rng.normal(0, noise, n_half)
    X_inner = np.stack([r_inner * np.cos(theta), r_inner * np.sin(theta)], axis=1)
    X_outer = np.stack([r_outer * np.cos(theta), r_outer * np.sin(theta)], axis=1)
    X = np.vstack([X_inner, X_outer])
    y = np.array([0]*n_half + [1]*n_half)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def make_moons(n=1000, noise=0.1, seed=42):
    """Two interleaving half-moons."""
    rng = np.random.default_rng(seed)
    n_half = n // 2
    theta0 = np.linspace(0, np.pi, n_half)
    theta1 = np.linspace(0, np.pi, n_half)
    X0 = np.stack([np.cos(theta0), np.sin(theta0)], axis=1)
    X1 = np.stack([1 - np.cos(theta1), 1 - np.sin(theta1) - 0.5], axis=1)
    X0 += rng.normal(0, noise, X0.shape)
    X1 += rng.normal(0, noise, X1.shape)
    X = np.vstack([X0, X1])
    y = np.array([0]*n_half + [1]*n_half)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def train_test_split_np(X, y, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    split = int(n * (1 - test_frac))
    tr, te = idx[:split], idx[split:]
    return X[tr], X[te], y[tr], y[te]


# ── Demos ──────────────────────────────────────────────────────

def demo_forward_backward():
    section("FORWARD + BACKWARD PASS — MANUAL TRACE")
    rng = np.random.default_rng(0)

    # Tiny 3-in, 2-hidden, 1-out network, 4 examples
    n_in, n_h, n_out, m = 3, 2, 1, 4
    net = TwoLayerNN(n_in, n_h, n_out, seed=0)

    X = rng.standard_normal((n_in, m))
    Y = rng.integers(0, 2, (1, m)).astype(float)

    A2 = net.forward(X)
    loss = net.compute_loss(Y)
    grads = net.backward(Y)

    print(f"  Input X shape:     {X.shape}  (n_features × m_examples)")
    print(f"  W1 shape:          {net.W1.shape}")
    print(f"  W2 shape:          {net.W2.shape}")
    print(f"\n  Z1 (pre-ReLU):     {net._cache['Z1'][:,0].round(4)}")
    print(f"  A1 (post-ReLU):    {net._cache['A1'][:,0].round(4)}")
    print(f"  Z2 (pre-sigmoid):  {net._cache['Z2'][0,:].round(4)}")
    print(f"  A2 (predictions):  {A2[0,:].round(4)}")
    print(f"  Y (labels):        {Y[0,:].astype(int)}")
    print(f"\n  BCE Loss:          {loss:.6f}")
    print(f"\n  dZ2 = A2 - Y:      {(A2 - Y)[0,:].round(4)}")
    print(f"  dW2:               {grads['dW2'].round(4)}")
    print(f"  dW1:               {grads['dW1'].round(4)}")


def demo_gradient_check():
    section("NUMERICAL GRADIENT CHECK")
    rng = np.random.default_rng(1)
    net = TwoLayerNN(4, 8, 1, seed=1)
    X = rng.standard_normal((4, 20))
    Y = rng.integers(0, 2, (1, 20)).astype(float)

    errors = numerical_gradient_check(net, X, Y, eps=1e-5)
    print("  Relative error between analytical and numerical gradients:")
    print("  (< 1e-5 confirms correct backprop implementation)")
    for key, err in errors.items():
        status = "PASS" if err < 1e-5 else "FAIL"
        print(f"  {key}: {err:.2e}  [{status}]")


def demo_initialization():
    section("WEIGHT INITIALIZATION COMPARISON")
    rng = np.random.default_rng(0)
    n_in, n_out = 512, 512
    n_layers = 10

    for name, init_fn in [
        ("Zero",   lambda: np.zeros((n_out, n_in))),
        ("Random large", lambda: rng.standard_normal((n_out, n_in)) * 1.0),
        ("Xavier", lambda: rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / (n_in + n_out))),
        ("He",     lambda: rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / n_in)),
    ]:
        a = rng.standard_normal((n_in, 100))
        vars_per_layer = []
        for _ in range(n_layers):
            W = init_fn()
            a = relu(W @ a)
            vars_per_layer.append(float(np.var(a)))
            if np.var(a) < 1e-20 or np.var(a) > 1e20:
                break

        if vars_per_layer[-1] < 1e-10:
            verdict = "VANISHING"
        elif vars_per_layer[-1] > 1e10:
            verdict = "EXPLODING"
        else:
            verdict = "STABLE"

        print(f"  {name:15s}: layer 1 var={vars_per_layer[0]:.4f}  "
              f"layer {len(vars_per_layer)} var={vars_per_layer[-1]:.2e}  → {verdict}")


def demo_train_circles():
    section("TRAINING ON CIRCLES DATASET — 2-LAYER NN")
    X, y = make_circles(n=1000, noise=0.05)
    X_tr, X_te, y_tr, y_te = train_test_split_np(X, y, test_frac=0.2)

    # Normalise
    mu, std = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr_n = ((X_tr - mu) / std).T   # (2, 800)
    X_te_n = ((X_te - mu) / std).T   # (2, 200)
    Y_tr = y_tr.reshape(1, -1)
    Y_te = y_te.reshape(1, -1)

    net = TwoLayerNN(n_input=2, n_hidden=16, n_output=1, seed=42)
    print(f"  Architecture: 2 → 16 (ReLU) → 1 (Sigmoid)")
    print(f"  Total params: {net.W1.size + net.b1.size + net.W2.size + net.b2.size}")
    history = train(net, X_tr_n, Y_tr, X_te_n, Y_te,
                    lr=0.05, n_epochs=500, batch_size=64, print_every=100)

    final = history[-1]
    print(f"\n  Final: loss={final['loss']:.4f}  "
          f"train_acc={final['train_acc']:.4f}  val_acc={final['val_acc']:.4f}")


def demo_train_deep():
    section("TRAINING ON MOONS DATASET — DEEP NN (3 HIDDEN LAYERS)")
    X, y = make_moons(n=1200, noise=0.1)
    X_tr, X_te, y_tr, y_te = train_test_split_np(X, y, test_frac=0.2)

    mu, std = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr_n = ((X_tr - mu) / std).T
    X_te_n = ((X_te - mu) / std).T
    Y_tr = y_tr.reshape(1, -1)
    Y_te = y_te.reshape(1, -1)

    dims = [2, 32, 16, 8, 1]
    net = DeepNN(dims, seed=7)
    n_params = sum(v.size for v in net.params.values())
    print(f"  Architecture: {' → '.join(str(d) for d in dims)}")
    print(f"  Total params: {n_params}")
    history = train(net, X_tr_n, Y_tr, X_te_n, Y_te,
                    lr=0.02, n_epochs=600, batch_size=64, print_every=150)

    final = history[-1]
    print(f"\n  Final: loss={final['loss']:.4f}  "
          f"train_acc={final['train_acc']:.4f}  val_acc={final['val_acc']:.4f}")


def demo_backprop_shapes():
    section("DIMENSION CHEAT SHEET — BACKPROP")
    print("""
  X:       (n_input,  m)     — input matrix
  W1:      (n_hidden, n_input)
  b1:      (n_hidden, 1)     — broadcast over m examples
  Z1:      (n_hidden, m)     = W1 @ X + b1
  A1:      (n_hidden, m)     = ReLU(Z1)
  W2:      (1,        n_hidden)
  b2:      (1,        1)
  Z2:      (1,        m)     = W2 @ A1 + b2
  A2:      (1,        m)     = sigmoid(Z2)
  Y:       (1,        m)     — labels

  Backprop:
  dZ2:     (1,        m)     = A2 - Y
  dW2:     (1,        n_hidden) = (dZ2 @ A1.T) / m
  db2:     (1,        1)     = mean(dZ2, axis=1)
  dA1:     (n_hidden, m)     = W2.T @ dZ2
  dZ1:     (n_hidden, m)     = dA1 * ReLU'(Z1)
  dW1:     (n_hidden, n_input)  = (dZ1 @ X.T) / m
  db1:     (n_hidden, 1)     = mean(dZ1, axis=1)
    """)


def main():
    demo_forward_backward()
    demo_gradient_check()
    demo_initialization()
    demo_backprop_shapes()
    demo_train_circles()
    demo_train_deep()


if __name__ == "__main__":
    main()
