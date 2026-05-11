# Project 05 — Neural Network Training Dashboard

> **Difficulty:** Intermediate · **Module:** 05 — Deep Learning & MLOps
> **Skills:** NumPy NN, Adam optimizer, cosine-warmup LR, MLflow, KS drift detection, early stopping

---

## What You'll Build

A 3-layer neural network trained on a 4-class spiral dataset (pure NumPy), with Adam + cosine-warmup schedule, early stopping (patience=10), MLflow experiment tracking, and KS-test drift detection on hidden activations. Outputs a training table + drift log to terminal.

---

## Skills Exercised

- 4-class spiral dataset generation (pure NumPy, no sklearn)
- Forward/backward pass for a 3-layer net with ReLU + softmax
- Adam optimizer with bias correction
- Cosine annealing with linear warmup
- MLflow: log loss, accuracy, LR, gradient norm per epoch
- KS test on activation distributions (hidden layer 1) across training
- Early stopping: track val_loss, patience counter, restore best weights

---

## Approach

### Phase 1 — Spiral dataset
```python
def make_spiral(n_per_class=200, n_classes=4, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        r = np.linspace(0.1, 1, n_per_class)
        theta = np.linspace(c*2*np.pi/n_classes,
                            (c+1)*2*np.pi/n_classes, n_per_class)
        theta += rng.normal(0, noise, n_per_class)
        X.append(np.column_stack([r*np.cos(theta), r*np.sin(theta)]))
        y.extend([c]*n_per_class)
    return np.vstack(X), np.array(y)
```

### Phase 2 — Network architecture
```
Input(2) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(4, Softmax)
Loss: cross-entropy
Parameters: W1(2×64), b1(64), W2(64×32), b2(32), W3(32×4), b3(4)
```

### Phase 3 — Training loop skeleton
```
for epoch in range(max_epochs):
    # forward pass
    z1 = X @ W1 + b1; a1 = relu(z1)
    z2 = a1 @ W2 + b2; a2 = relu(z2)
    z3 = a2 @ W3 + b3; a3 = softmax(z3)
    loss = cross_entropy(y_onehot, a3)

    # backward pass (derive deltas, compute grads)
    # Adam update for each parameter
    # LR schedule: cosine warmup
    # early stopping check on val_loss
    # MLflow log_metric(loss, acc, lr, grad_norm)
    # KS test on a1 vs baseline_a1 (activations at epoch 0)
    # if ks.pvalue < 0.05: log drift event
```

### Phase 4 — Cosine warmup schedule
```python
def lr_schedule(epoch, warmup=10, total=200, lr_max=1e-3, lr_min=1e-5):
    if epoch < warmup:
        return lr_max * epoch / warmup
    progress = (epoch - warmup) / (total - warmup)
    return lr_min + 0.5*(lr_max - lr_min)*(1 + np.cos(np.pi*progress))
```

### Phase 5 — Output table
```
Epoch |  LR      | Train Loss | Val Loss | Val Acc | Grad Norm | KS p-val
  10  | 1.00e-3  |   1.382    |  1.396   |  25.0%  |   0.041   |  0.821
  20  | 9.78e-4  |   1.201    |  1.234   |  51.2%  |   0.038   |  0.412
...
EARLY STOP at epoch 87, best val_loss=0.312 at epoch 77
```

---

## Checkpoints

| Phase | What correct output looks like |
|-------|-------------------------------|
| 1 | `X.shape == (800, 2)`, `np.unique(y) == [0,1,2,3]`; 4 interleaved spirals visually (if plotted) |
| 2 | `a3.sum(axis=1)` → all 1.0 (softmax); `loss` at epoch 0 ≈ `log(4) ≈ 1.386` |
| 3 | Val accuracy > 90% by epoch 150 on non-noisy spiral |
| 4 | LR at epoch 0 = 0; peaks at epoch 10; decreases toward `lr_min` |
| 5 | MLflow run visible in `mlflow ui`; drift events logged when activations shift |

---

## Extensions

1. **Adam vs SGD-Momentum comparison** — run both optimizers, log to separate MLflow experiments, print final val_acc comparison.
2. **Permutation feature importance** — after training, shuffle each input feature independently 10 times, measure accuracy drop; print importance table.
3. **Batch normalization** — add a BatchNorm layer between Dense and ReLU; compare convergence speed and final accuracy with/without.

---

## Hints

<details><summary>Hint 1 — Softmax numerical stability</summary>
<code>e = np.exp(z - z.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)</code>. Subtract the row max before exp.
</details>

<details><summary>Hint 2 — Cross-entropy backward into softmax</summary>
For softmax + cross-entropy combined: <code>dz3 = a3 - y_onehot</code> (the delta for z3 directly). This is the fused gradient; no need to backprop through softmax separately.
</details>

<details><summary>Hint 3 — Adam bias correction</summary>
<code>m_hat = m / (1 - beta1**t); v_hat = v / (1 - beta2**t)</code> where t is the step count. Without this, early updates are too small.
</details>

<details><summary>Hint 4 — Early stopping — save a copy of weights</summary>
<code>best_weights = {k: v.copy() for k,v in params.items()}</code> when val_loss improves. Restore after patience exhausted.
</details>

<details><summary>Hint 5 — KS test drift baseline</summary>
Save <code>baseline_activations = a1.flatten()</code> at epoch 0. Each epoch: <code>scipy.stats.ks_2samp(baseline_activations, a1.flatten())</code>. A small p-value (< 0.05) means the activation distribution has shifted significantly.
</details>

---

*Back to [Module 05 — Deep Learning & MLOps](../05-deep-learning.md)*
