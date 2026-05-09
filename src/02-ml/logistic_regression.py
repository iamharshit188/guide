import numpy as np


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Activation & loss ─────────────────────────────────────────

def sigmoid(z):
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))   # numerically stable


def softmax(Z):
    """Z: (N, K). Numerically stable via max subtraction."""
    Z_shifted = Z - Z.max(axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


def binary_cross_entropy(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def categorical_cross_entropy(Y_onehot, P, eps=1e-12):
    return -np.mean(np.sum(Y_onehot * np.log(np.clip(P, eps, 1)), axis=1))


# ── Binary Logistic Regression ────────────────────────────────

class LogisticRegression:
    """Binary logistic regression with L2 regularization, trained via GD."""

    def __init__(self, lr=0.1, n_iters=1000, l2=0.0):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.losses = []

        for _ in range(self.n_iters):
            z = X @ self.w + self.b
            p = sigmoid(z)
            loss = binary_cross_entropy(y, p) + 0.5 * self.l2 * np.dot(self.w, self.w)
            self.losses.append(loss)

            # Gradient: ∇_w L = (1/N) X^T (p - y) + λw
            error = p - y
            grad_w = (X.T @ error) / n + self.l2 * self.w
            grad_b = np.mean(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ── Softmax Regression (Multiclass) ──────────────────────────

class SoftmaxRegression:
    """K-class softmax regression via GD."""

    def __init__(self, lr=0.1, n_iters=500):
        self.lr = lr
        self.n_iters = n_iters
        self.W = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        n, d = X.shape
        K = len(np.unique(y))
        Y = np.eye(K)[y]  # one-hot (N, K)

        self.W = np.zeros((d, K))
        self.b = np.zeros(K)
        self.losses = []

        for _ in range(self.n_iters):
            Z = X @ self.W + self.b  # (N, K)
            P = softmax(Z)
            loss = categorical_cross_entropy(Y, P)
            self.losses.append(loss)

            # Gradient: ∇_W L = X^T (P - Y) / N
            dZ = (P - Y) / n
            self.W -= self.lr * X.T @ dZ
            self.b -= self.lr * dZ.sum(axis=0)

        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ── Metrics ───────────────────────────────────────────────────

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, K=2):
    C = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    return C


def precision_recall_f1(y_true, y_pred, positive=1):
    tp = np.sum((y_pred == positive) & (y_true == positive))
    fp = np.sum((y_pred == positive) & (y_true != positive))
    fn = np.sum((y_pred != positive) & (y_true == positive))
    prec = tp / (tp + fp + 1e-10)
    rec  = tp / (tp + fn + 1e-10)
    f1   = 2 * prec * rec / (prec + rec + 1e-10)
    return prec, rec, f1


def roc_auc(y_true, y_score):
    """Compute AUC-ROC via trapezoidal rule."""
    thresholds = np.sort(np.unique(y_score))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        tprs.append(tp / pos)
        fprs.append(fp / neg)
    tprs.append(1.0); fprs.append(1.0)
    tprs, fprs = np.array(tprs), np.array(fprs)
    return np.trapz(tprs, fprs)


# ── Datasets ─────────────────────────────────────────────────

def make_binary_dataset(n=400, noise=0.15, seed=42):
    rng = np.random.default_rng(seed)
    X0 = rng.normal([-1, -1], 0.6, (n//2, 2))
    X1 = rng.normal([1, 1], 0.6, (n//2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0]*(n//2) + [1]*(n//2))
    # Add noise
    flip_idx = rng.choice(n, int(noise*n), replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    return X, y


def make_multiclass_dataset(n=300, seed=0):
    rng = np.random.default_rng(seed)
    centers = [[-2, -2], [2, -2], [0, 2.5]]
    X = np.vstack([rng.normal(c, 0.7, (n//3, 2)) for c in centers])
    y = np.array([k for k in range(3) for _ in range(n//3)])
    return X, y


def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_size))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


# ── Main ──────────────────────────────────────────────────────

def main():
    section("1. BINARY LOGISTIC REGRESSION FROM SCRATCH")

    X, y = make_binary_dataset(n=500)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    # Standardize
    mu, std = X_tr.mean(0), X_tr.std(0)
    X_tr_s = (X_tr - mu) / std
    X_te_s = (X_te - mu) / std

    model = LogisticRegression(lr=0.5, n_iters=300, l2=0.01)
    model.fit(X_tr_s, y_tr)

    y_pred = model.predict(X_te_s)
    y_proba = model.predict_proba(X_te_s)

    prec, rec, f1 = precision_recall_f1(y_te, y_pred)
    auc = roc_auc(y_te, y_proba)

    print(f"Learned weights: {model.w.round(4)}, bias: {model.b:.4f}")
    print(f"Loss: initial={model.losses[0]:.4f} → final={model.losses[-1]:.4f}")
    print(f"Accuracy:  {accuracy(y_te, y_pred):.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

    cm = confusion_matrix(y_te, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    section("2. THRESHOLD ANALYSIS")
    print("Varying decision threshold:")
    print(f"{'Threshold':>10}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}  {'Acc':>8}")
    print("-" * 50)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_t = (y_proba >= t).astype(int)
        p, r, f = precision_recall_f1(y_te, preds_t)
        a = accuracy(y_te, preds_t)
        print(f"{t:>10.1f}  {p:>8.4f}  {r:>8.4f}  {f:>8.4f}  {a:>8.4f}")

    section("3. REGULARIZATION COMPARISON (L2)")
    print(f"{'L2 λ':>10}  {'Train Loss':>12}  {'Test Acc':>10}  {'||w||':>8}")
    print("-" * 45)
    for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
        m = LogisticRegression(lr=0.5, n_iters=300, l2=lam).fit(X_tr_s, y_tr)
        acc_te = accuracy(y_te, m.predict(X_te_s))
        print(f"{lam:>10.3f}  {m.losses[-1]:>12.4f}  {acc_te:>10.4f}  {np.linalg.norm(m.w):>8.4f}")

    section("4. SOFTMAX REGRESSION — MULTICLASS")
    X3, y3 = make_multiclass_dataset(n=450)
    X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, y3)
    mu3, std3 = X3_tr.mean(0), X3_tr.std(0)
    X3_tr_s = (X3_tr - mu3) / std3
    X3_te_s = (X3_te - mu3) / std3

    sm = SoftmaxRegression(lr=1.0, n_iters=500).fit(X3_tr_s, y3_tr)
    y3_pred = sm.predict(X3_te_s)

    print(f"3-class problem: classes={np.unique(y3_te)}, N_test={len(y3_te)}")
    print(f"Loss: {sm.losses[0]:.4f} → {sm.losses[-1]:.4f}")
    print(f"Test Accuracy: {accuracy(y3_te, y3_pred):.4f}")
    print(f"Weight matrix W shape: {sm.W.shape}  (d=2, K=3)")
    print(f"\nPer-class accuracy:")
    for k in range(3):
        mask = y3_te == k
        acc_k = np.mean(y3_pred[mask] == k)
        print(f"  Class {k}: {acc_k:.4f}  ({mask.sum()} samples)")

    section("5. SIGMOID PROPERTIES")
    z_vals = np.array([-5, -2, -1, 0, 1, 2, 5], dtype=float)
    sig_vals = sigmoid(z_vals)
    deriv_vals = sig_vals * (1 - sig_vals)
    print(f"{'z':>6}  {'σ(z)':>10}  {'σ(z)(1-σ(z))':>14}")
    print("-" * 35)
    for z, s, d in zip(z_vals, sig_vals, deriv_vals):
        print(f"{z:>6.1f}  {s:>10.4f}  {d:>14.4f}")
    print(f"\nLog-odds: log(p/(1-p)) = w^Tx")
    for z, s in zip([-2, 0, 2], sigmoid(np.array([-2.0, 0.0, 2.0]))):
        odds = s / (1 - s) if s < 1 else float('inf')
        print(f"  z={z}: p={s:.3f}, odds={odds:.3f}, log-odds={z}")


if __name__ == "__main__":
    main()
