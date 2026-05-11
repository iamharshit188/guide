# Project 01 — PCA Image Compressor

> **Difficulty:** Beginner · **Module:** 01 — Math for ML
> **Skills:** NumPy, eigendecomposition, linear algebra, reconstruction error

---

## What You'll Build

A command-line tool that loads a grayscale image as a NumPy matrix, compresses it using PCA at varying ranks, and prints a table of RMSE vs. explained variance for each rank. No sklearn in the core pipeline — eigendecomposition only.

---

## Skills Exercised

- Centering data: $X_c = X - \bar{X}$
- Covariance matrix: $C = \frac{1}{n-1}X_c^T X_c$
- Eigendecomposition via `np.linalg.eigh`
- Projection and reconstruction: $\hat{X} = X_c V_k V_k^T + \bar{X}$
- Reconstruction RMSE and explained variance ratio (EVR)

---

## Approach

### Phase 1 — Load and prep data
```
load grayscale image → float64 → shape (H, W)
flatten to (H, W) → treat each row as a sample, columns as features
center: subtract column means
```

### Phase 2 — Eigendecomposition
```
compute C = (1/(n-1)) * X_c.T @ X_c    # shape (W, W)
eigenvalues, eigenvectors = np.linalg.eigh(C)
sort descending by eigenvalue magnitude
```

### Phase 3 — Project and reconstruct at multiple ranks
```
for rank in [5, 10, 20, 50]:
    V_k = top-k eigenvectors           # shape (W, k)
    scores = X_c @ V_k                 # (H, k)
    X_hat = scores @ V_k.T + col_means # (H, W) → reshape to (H, W)
    rmse = sqrt(mean((X - X_hat)**2))
    evr  = sum(top-k eigenvalues) / sum(all eigenvalues)
    print row in table
```

### Phase 4 — Output table
```
Rank  |  RMSE   |  EVR (%)  |  Compression ratio
  5   |  ...    |  ...      |  W*H / (H*5 + W*5)
 10   |  ...    |  ...      |  ...
 20   |  ...    |  ...      |  ...
 50   |  ...    |  ...      |  ...
```

---

## Checkpoints

| Phase | What correct output looks like |
|-------|-------------------------------|
| 1 | `X_c.mean(axis=0)` is all zeros; `X_c.shape == (H, W)` |
| 2 | Eigenvalues in descending order; `eigenvectors.shape == (W, W)` |
| 3 | `X_hat` pixel values in [0, 255]; RMSE decreases as rank increases |
| 4 | EVR at rank 50 should exceed 90% for most natural images |

---

## Extensions

1. **Face dataset** — use `sklearn.datasets.fetch_olivetti_faces` (64×64 grayscale). Compare EVR curves across subjects.
2. **Incremental PCA** — implement randomized SVD (`np.linalg.svd` with `full_matrices=False`) instead of full eigendecomposition; compare speed on 512×512 images.
3. **Color images** — apply PCA independently to R, G, B channels; merge reconstructed channels.

---

## Hints

<details><summary>Hint 1 — Why eigh instead of eig?</summary>
<code>np.linalg.eigh</code> is for symmetric/Hermitian matrices (which C always is). It returns real eigenvalues and is more numerically stable than <code>eig</code>.
</details>

<details><summary>Hint 2 — Sort order</summary>
<code>np.linalg.eigh</code> returns eigenvalues in ascending order. Reverse with <code>[::-1]</code> on both eigenvalues and eigenvectors columns.
</details>

<details><summary>Hint 3 — Reconstruction must add back the mean</summary>
<code>X_hat = scores @ V_k.T + col_means</code>. Forgetting this causes the reconstructed image to appear mean-subtracted (gray, low contrast).
</details>

<details><summary>Hint 4 — Compression ratio formula</summary>
Original storage: H×W floats. Compressed storage: H×k (scores) + W×k (basis) = k×(H+W). Ratio = H×W / (k×(H+W)).
</details>

<details><summary>Hint 5 — No image file? Generate synthetic data</summary>
<code>rng = np.random.default_rng(42); X = rng.integers(0, 256, (128, 128), dtype=float)</code> works fine for testing the pipeline.
</details>

---

*Back to [Module 01 — Math for ML](../01-math.md)*
