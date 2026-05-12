# Module 01 — Math for ML

> **Runnable code:** `src/01-math/`
>
> ```bash
> python src/01-math/vectors.py
> python src/01-math/matrix_ops.py
> python src/01-math/calculus_demo.py
> python src/01-math/probability.py
> ```

---

## Prerequisites & Overview

**Prerequisites:** High school algebra, basic Python, knowing what a `for` loop is. No prior ML knowledge needed.
**Estimated time:** 8–12 hours (reading + running all scripts + mini-project at the end)

**Install dependencies first:**
```bash
pip install numpy
```

---

## Why This Module Matters

Every modern AI system is built on three mathematical pillars:

| Pillar | What it does in ML | Example |
|--------|-------------------|---------|
| **Linear Algebra** | Represents and transforms data | Neural network layers are matrix multiplications |
| **Calculus** | Tells the model how to improve | Gradient descent minimizes error using derivatives |
| **Probability** | Handles uncertainty and noise | Classifier outputs a probability, not a hard label |

Without this foundation, ML becomes memorizing APIs. With it, you can debug broken training, understand why attention works, derive new architectures, and answer senior-level interview questions.

---

## How to Read This Guide

Every concept follows this pattern:

1. **Real-world analogy** — anchor to something you already know
2. **Math formula** — see the precise definition
3. **Tiny numeric example** — trace through with small numbers by hand
4. **Python code** — run it, read the output, modify it
5. **ML connection** — where exactly this appears in real systems

---

# PART 1 — Linear Algebra

Linear algebra is the language of ML. Almost every computation — from a single neuron to a billion-parameter LLM — is linear algebra under the hood.

---

## 1.1 Scalars, Vectors, Matrices, Tensors

### Intuition

Think of organizing information about students:
- **Scalar**: one number — "Alice scored 92 on the math test"
- **Vector**: one student's full profile — `[math=92, physics=88, attendance=95]`
- **Matrix**: entire classroom — stack each student's vector as a row
- **Tensor**: batch of classrooms across multiple schools — another dimension on top

### Formal Definitions

| Object | Notation | Shape | Example |
|--------|----------|-------|---------|
| Scalar | $x \in \mathbb{R}$ | `()` | `3.14` |
| Vector | $\mathbf{v} \in \mathbb{R}^n$ | `(n,)` | `[1, 2, 3]` |
| Matrix | $A \in \mathbb{R}^{m \times n}$ | `(m, n)` | `[[1,2],[3,4]]` |
| Tensor | $\mathcal{T}$ | `(d1, d2, ..., dk)` | batch of images |

### Python Code — Creating Each Object

```python
import numpy as np

# ── SCALAR ──────────────────────────────────────────────────
score = 92.0
print(f"Scalar: {score}")           # 92.0
print(f"Shape:  {np.shape(score)}") # ()  ← zero-dimensional

# ── VECTOR ──────────────────────────────────────────────────
# Alice's profile: [math_score, physics_score, attendance%]
alice = np.array([92, 88, 95])
print(f"\nVector: {alice}")         # [92 88 95]
print(f"Shape:  {alice.shape}")     # (3,)  ← 3 elements

# ── MATRIX ──────────────────────────────────────────────────
# Entire classroom — each row is one student
classroom = np.array([
    [92, 88, 95],   # Alice
    [75, 81, 90],   # Bob
    [84, 79, 91],   # Charlie
])
print(f"\nMatrix:\n{classroom}")
print(f"Shape: {classroom.shape}")  # (3, 3) ← 3 students, 3 features

# ── TENSOR ──────────────────────────────────────────────────
# 4 schools, each with 3 students, each with 3 features
schools = np.random.randint(60, 100, size=(4, 3, 3))
print(f"\nTensor shape: {schools.shape}")  # (4, 3, 3)

# Accessing elements:
print(f"\nAlice's physics score: {classroom[0, 1]}")  # 88
print(f"All physics scores:     {classroom[:, 1]}")   # [88 81 79]
```

**What each line does:**
- `np.array([...])` — wraps a Python list into a NumPy array that can do math
- `alice.shape` — tells you the dimensions; `(3,)` means 1D with 3 elements
- `classroom[:, 1]` — `:` means "all rows", `1` means "column index 1" (physics)

**ML connection:** When you pass a batch of 32 images through a CNN, each image is a `(3, 224, 224)` tensor (RGB channels × height × width). The batch becomes `(32, 3, 224, 224)`. Every layer transforms this 4D tensor.

---

## 1.2 Vector Operations

### Intuition

Vectors are arrows in space. Operations on vectors are geometric operations — adding arrows, stretching them, measuring angles between them.

### Vector Addition

$$\mathbf{u} + \mathbf{v} = [u_1 + v_1,\ u_2 + v_2,\ \ldots,\ u_n + v_n]$$

**Real-world:** Alice's week 1 scores `[80, 70]` + improvement `[12, 18]` = final scores `[92, 88]`.

```python
import numpy as np

week1  = np.array([80, 70])
gains  = np.array([12, 18])
final  = week1 + gains

print(f"Week 1:      {week1}")   # [80 70]
print(f"Improvement: {gains}")   # [12 18]
print(f"Final:       {final}")   # [92 88]
```

### Scalar Multiplication

$$c \cdot \mathbf{v} = [c v_1,\ c v_2,\ \ldots,\ c v_n]$$

**Real-world:** Double all scores for extra credit.

```python
scores  = np.array([40, 35, 48])
doubled = 2 * scores
print(f"Original: {scores}")   # [40 35 48]
print(f"Doubled:  {doubled}")  # [80 70 96]
```

### Dot Product (Inner Product)

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = \mathbf{u}^T \mathbf{v}$$

**Step-by-step numeric example:**
$$[1, 2, 3] \cdot [4, 5, 6] = 1(4) + 2(5) + 3(6) = 4 + 10 + 18 = 32$$

```python
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Method 1: manual
manual = u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

# Method 2: NumPy
dot = np.dot(u, v)

# Method 3: @ operator (same as dot for 1D)
dot2 = u @ v

print(f"Manual:  {manual}")  # 32
print(f"np.dot:  {dot}")     # 32
print(f"@ op:    {dot2}")    # 32
```

**Geometric meaning:**
$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

| Angle | $\cos\theta$ | Dot product | Meaning |
|-------|-------------|-------------|---------|
| $0°$ | 1 | Maximum positive | Vectors point the same way |
| $90°$ | 0 | Zero | Vectors are perpendicular (independent) |
| $180°$ | -1 | Maximum negative | Vectors point opposite ways |

```python
# Dot product and angle
import numpy as np

a = np.array([1, 0])  # points right
b = np.array([0, 1])  # points up

dot_ab = np.dot(a, b)
cos_theta = dot_ab / (np.linalg.norm(a) * np.linalg.norm(b))
angle_deg = np.degrees(np.arccos(cos_theta))

print(f"Dot product: {dot_ab}")       # 0
print(f"Angle:       {angle_deg}°")   # 90.0° — perpendicular
```

**ML connection:** In a recommender system, the user embedding `u` and movie embedding `m` are dotted. Higher dot product → stronger predicted preference. In Transformers, attention scores are dot products between query and key vectors.

---

## 1.3 Norms — Measuring Vector Size

### Intuition

A norm is a ruler for vectors. Different norms measure "size" differently, like measuring a city block by direct distance vs. walking along streets.

### p-Norm

$$\|\mathbf{v}\|_p = \left(\sum_{i=1}^{n} |v_i|^p\right)^{1/p}$$

### The Three Norms You Must Know

| Norm | Formula | Intuition | ML use |
|------|---------|-----------|--------|
| L1 | $\|\mathbf{v}\|_1 = \sum_i |v_i|$ | Manhattan distance (taxi cab) | Lasso, sparse models |
| L2 | $\|\mathbf{v}\|_2 = \sqrt{\sum_i v_i^2}$ | Euclidean distance (straight line) | Ridge, weight decay, cosine sim |
| L∞ | $\|\mathbf{v}\|_\infty = \max_i |v_i|$ | Largest component | Adversarial robustness |

```python
import numpy as np

v = np.array([3.0, -4.0, 0.0])

# L1 norm: sum of absolute values
l1 = np.linalg.norm(v, ord=1)
print(f"L1: |3| + |-4| + |0| = {l1}")  # 7.0

# L2 norm: Euclidean length
l2 = np.linalg.norm(v, ord=2)  # or just np.linalg.norm(v)
print(f"L2: sqrt(9 + 16 + 0) = {l2}")  # 5.0

# L-inf norm: largest absolute value
linf = np.linalg.norm(v, ord=np.inf)
print(f"L∞: max(3, 4, 0) = {linf}")    # 4.0

# Why L1 promotes sparsity:
# When you penalize ||w||_1, the optimizer prefers weights at exactly zero
# because reducing a non-zero weight to zero gives a fixed decrease in the penalty
# no matter how small the weight was.
```

**L1 vs L2 — the key difference:**
- **L1** penalizes weights equally regardless of size → pushes small weights to exactly zero → sparsity
- **L2** penalizes large weights more → shrinks all weights smoothly → no zeros

### Cosine Similarity

$$\text{cosine\_sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \in [-1, 1]$$

Cosine similarity measures **direction** only, ignoring magnitude. Two vectors pointing the same direction have similarity 1 even if one is 10× longer.

```python
import numpy as np

def cosine_similarity(u, v):
    # Step 1: compute the dot product
    dot = np.dot(u, v)

    # Step 2: compute the L2 norm of each vector
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    # Step 3: divide dot by product of norms
    return dot / (norm_u * norm_v)

# Example: two students with similar profiles (same direction)
alice   = np.array([92, 88, 95])
charlie = np.array([46, 44, 47.5])  # exactly half of Alice's scores

sim1 = cosine_similarity(alice, charlie)
print(f"Alice vs Charlie (same direction): {sim1:.4f}")  # 1.0

# Example: two students with very different profiles
bob  = np.array([75, 81, 90])
dave = np.array([20, 95, 50])

sim2 = cosine_similarity(bob, dave)
print(f"Bob vs Dave (different profiles): {sim2:.4f}")  # ~0.91

# Normalizing first (unit vectors)
alice_unit = alice / np.linalg.norm(alice)
print(f"Alice unit vector: {alice_unit.round(4)}")
# After normalization, dot product = cosine similarity
```

**ML connection:** Semantic search — when you query "what is backpropagation?", the system embeds your query into a vector and finds document embeddings with the highest cosine similarity. ChromaDB, Pinecone, and FAISS all do this under the hood.

---

## 1.4 Matrix Operations

### Matrix Multiplication

If $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$, then $C = AB \in \mathbb{R}^{m \times n}$ where:

$$C_{ij} = \sum_{l=1}^{k} A_{il} B_{lj}$$

**Dimension rule:** Inner dimensions must match. `(m × k) × (k × n) → (m × n)`.

**Step-by-step example:**
$$\begin{bmatrix}1&2\\3&4\end{bmatrix} \times \begin{bmatrix}5&6\\7&8\end{bmatrix}$$

Row 0 × Col 0: $1(5) + 2(7) = 19$
Row 0 × Col 1: $1(6) + 2(8) = 22$
Row 1 × Col 0: $3(5) + 4(7) = 43$
Row 1 × Col 1: $3(6) + 4(8) = 50$

Result: $\begin{bmatrix}19&22\\43&50\end{bmatrix}$

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Method 1: np.dot
C = np.dot(A, B)

# Method 2: @ operator (preferred for matrices)
C2 = A @ B

print(f"A @ B =\n{C}")
# [[19 22]
#  [43 50]]

# Verify manually
c00 = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]  # = 1*5 + 2*7 = 19
print(f"C[0,0] manually: {c00}")

# Shape check
print(f"A: {A.shape}, B: {B.shape}, C: {C.shape}")  # (2,2) × (2,2) = (2,2)

# Batched matrix multiply (crucial for ML)
# Input batch: 32 samples, each with 10 features
X = np.random.randn(32, 10)
# Weight matrix: 10 input features → 5 output features
W = np.random.randn(10, 5)
# Output: 32 samples, each with 5 activations
out = X @ W
print(f"\nNeural layer: {X.shape} @ {W.shape} = {out.shape}")
# (32, 10) @ (10, 5) = (32, 5)
```

**Key properties:**

```python
# Associative: (AB)C = A(BC)
A = np.random.randn(2, 3)
B = np.random.randn(3, 4)
C = np.random.randn(4, 2)

left  = (A @ B) @ C
right = A @ (B @ C)
print(f"Associative holds: {np.allclose(left, right)}")  # True

# NOT commutative: AB ≠ BA in general
M = np.array([[1,2],[3,4]])
N = np.array([[0,1],[1,0]])
print(f"\nM@N =\n{M@N}")   # [[2,1],[4,3]]
print(f"N@M =\n{N@M}")    # [[3,4],[1,2]]  ← different!
```

**ML connection:** Every dense layer in a neural network is `out = X @ W + b`. A batch of 64 inputs through a hidden layer of 512 units is literally a `(64, input_dim) @ (input_dim, 512)` matrix multiplication.

### Transpose

$$A^T_{ij} = A_{ji}$$

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(f"A shape: {A.shape}")      # (2, 3)
print(f"A.T shape: {A.T.shape}")  # (3, 2)
print(f"A.T:\n{A.T}")
# [[1 4]
#  [2 5]
#  [3 6]]

# Key identity: (AB)^T = B^T A^T
B = np.random.randn(3, 4)
AB_T   = (A @ B).T
BT_AT  = B.T @ A.T
print(f"\n(AB)^T = B^T A^T: {np.allclose(AB_T, BT_AT)}")  # True
```

### Trace and Determinant

```python
M = np.array([[4, 2],
              [1, 3]])

# Trace: sum of diagonal elements
trace = np.trace(M)
print(f"Trace: {trace}")          # 7  (4 + 3)

# Determinant: measures volume scaling / invertibility
det = np.linalg.det(M)
print(f"Det: {det}")              # 10.0  (4*3 - 2*1)

# If det = 0, matrix is singular (non-invertible)
singular = np.array([[1, 2], [2, 4]])
print(f"Singular det: {np.linalg.det(singular):.6f}")  # 0.000000
```

---

## 1.5 Matrix Inverse

For a square matrix $A$:
$$A A^{-1} = I$$

where $I$ is the identity matrix.

**Why inverses matter:** Solving $A\mathbf{x} = \mathbf{b}$ gives $\mathbf{x} = A^{-1}\mathbf{b}$.

```python
import numpy as np

A = np.array([[4.0, 2.0],
              [1.0, 3.0]])

A_inv = np.linalg.inv(A)
print(f"A inverse:\n{A_inv}")
# [[ 0.3  -0.2]
#  [-0.1   0.4]]

# Verify: A @ A_inv should be identity
I_approx = A @ A_inv
print(f"\nA @ A_inv:\n{I_approx.round(10)}")
# [[1. 0.]
#  [0. 1.]]

# Solving Ax = b using inverse
b = np.array([8.0, 5.0])
x = A_inv @ b
print(f"\nSolution x: {x}")         # [1.4 1.8]
print(f"Verify Ax=b: {A @ x}")     # [8. 5.] ✓

# Moore-Penrose pseudoinverse (works for non-square / singular matrices)
# Used in linear regression closed-form: w = (X^T X)^-1 X^T y
A_rect = np.random.randn(5, 3)     # 5 equations, 3 unknowns
A_pinv = np.linalg.pinv(A_rect)   # shape (3, 5)
print(f"\nPseudoinverse shape: {A_pinv.shape}")
```

---

## 1.6 Eigenvalues & Eigenvectors

### Intuition

Every matrix is a transformation — it rotates and stretches vectors. Eigenvectors are the special vectors that **only get stretched, never rotated**. The eigenvalue says by how much.

$$A\mathbf{v} = \lambda\mathbf{v}$$

- $\mathbf{v}$ = eigenvector (direction preserved)
- $\lambda$ = eigenvalue (how much it scales)

### Why This Matters for ML

PCA finds the directions (eigenvectors) of maximum variance in the data. The eigenvalue of each direction tells you how much variance lies along it. The top eigenvectors become your compressed representation.

```python
import numpy as np

# Example: 2x2 matrix
A = np.array([[3.0, 1.0],
              [1.0, 3.0]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues:  {eigenvalues}")     # [4. 2.]
print(f"Eigenvectors:\n{eigenvectors}")
# Each COLUMN is one eigenvector

# Verify: A @ v = λ @ v
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v   = eigenvectors[:, i]  # i-th column

    Av     = A @ v
    lam_v  = lam * v

    print(f"\nEigenvector {i}: {v.round(4)}")
    print(f"  A @ v    = {Av.round(4)}")
    print(f"  λ × v    = {lam_v.round(4)}")
    print(f"  Match:     {np.allclose(Av, lam_v)}")

# Eigendecomposition: A = V Λ V^-1
V      = eigenvectors
Lambda = np.diag(eigenvalues)
A_reconstructed = V @ Lambda @ np.linalg.inv(V)
print(f"\nReconstructed A matches original: {np.allclose(A, A_reconstructed)}")
```

**Step-by-step hand calculation:**

Characteristic equation: $\det(A - \lambda I) = 0$

$$\det\begin{bmatrix}3-\lambda & 1 \\ 1 & 3-\lambda\end{bmatrix} = (3-\lambda)^2 - 1 = 0$$
$$\lambda^2 - 6\lambda + 8 = 0 \Rightarrow (\lambda-4)(\lambda-2) = 0$$
$$\lambda_1 = 4,\quad \lambda_2 = 2$$

### PCA Preview with Eigendecomposition

```python
import numpy as np

rng = np.random.default_rng(42)

# Simulate correlated student data
n = 100
math_score = rng.normal(75, 10, n)
physics_score = math_score * 0.8 + rng.normal(0, 5, n)  # correlated

X = np.column_stack([math_score, physics_score])
X -= X.mean(axis=0)  # center the data

# Covariance matrix — captures spread and correlation
cov = X.T @ X / n
print(f"Covariance matrix:\n{cov.round(2)}")

# Find principal components
eigenvalues, eigenvectors = np.linalg.eig(cov)

# Sort by largest eigenvalue (most variance first)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Explained variance ratio
total_var = eigenvalues.sum()
print(f"\nVariance explained by PC1: {eigenvalues[0]/total_var*100:.1f}%")
print(f"Variance explained by PC2: {eigenvalues[1]/total_var*100:.1f}%")

# Project onto first principal component (dimensionality reduction)
X_compressed = X @ eigenvectors[:, 0:1]
print(f"\nOriginal shape: {X.shape}")          # (100, 2)
print(f"Compressed shape: {X_compressed.shape}")  # (100, 1)
```

---

## 1.7 Singular Value Decomposition (SVD)

Any matrix $A \in \mathbb{R}^{m \times n}$ (square or not) decomposes as:

$$A = U \Sigma V^T$$

| Matrix | Shape | Meaning |
|--------|-------|---------|
| $U$ | $(m, m)$ | Left singular vectors — row space basis |
| $\Sigma$ | $(m, n)$ diagonal | Singular values — importance of each dimension |
| $V^T$ | $(n, n)$ | Right singular vectors — column space basis |

```python
import numpy as np

# User-movie rating matrix (4 users, 5 movies)
A = np.array([
    [5, 4, 1, 1, 2],
    [4, 5, 1, 2, 1],
    [1, 2, 5, 4, 4],
    [2, 1, 4, 5, 4],
], dtype=float)

# Full SVD
U, sigma, Vt = np.linalg.svd(A, full_matrices=True)
print(f"U shape:     {U.shape}")      # (4, 4)
print(f"Sigma:       {sigma.round(2)}")  # singular values, largest first
print(f"Vt shape:    {Vt.shape}")     # (5, 5)

# Reconstruct original matrix from all singular values
Sigma_full = np.zeros_like(A)
np.fill_diagonal(Sigma_full, sigma)
A_reconstructed = U @ Sigma_full @ Vt
print(f"\nReconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")

# ── Truncated SVD (the useful part) ────────────────────────
k = 2  # keep only top 2 singular values

U_k     = U[:, :k]           # (4, 2)
sigma_k = np.diag(sigma[:k]) # (2, 2)
Vt_k    = Vt[:k, :]          # (2, 5)

A_approx = U_k @ sigma_k @ Vt_k
print(f"\nWith k={k} components:")
print(f"Approximation error: {np.linalg.norm(A - A_approx):.4f}")
print(f"Variance captured: {(sigma[:k]**2).sum() / (sigma**2).sum() * 100:.1f}%")

# User "latent features" for recommendation
user_embeddings = U_k @ sigma_k
print(f"\nUser latent embeddings (4 users in 2D):\n{user_embeddings.round(2)}")
```

**ML connection:** SVD powers recommendation systems (collaborative filtering), NLP (Latent Semantic Analysis), image compression, and is the foundation of PCA. Modern large model training uses low-rank SVD decompositions (LoRA is essentially truncated SVD applied to weight updates).

---

# PART 2 — Calculus for ML

Calculus is how models learn. Without derivatives, there is no backpropagation, and without backpropagation, neural networks cannot update their weights.

---

## 2.1 Derivatives — The Core Idea

### Intuition

A derivative tells you the **slope at a point**: if you change the input by a tiny amount, how much does the output change?

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Analogy:** You're driving uphill. The derivative of elevation with respect to position tells you how steep the current section is.

### Common Derivatives You Must Know

| Function | Derivative |
|----------|-----------|
| $f(x) = c$ (constant) | $f'(x) = 0$ |
| $f(x) = x^n$ | $f'(x) = nx^{n-1}$ |
| $f(x) = e^x$ | $f'(x) = e^x$ |
| $f(x) = \ln(x)$ | $f'(x) = 1/x$ |
| $f(x) = \sigma(x)$ (sigmoid) | $f'(x) = \sigma(x)(1 - \sigma(x))$ |

```python
import numpy as np

# Numerical derivative — approximate using tiny step h
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)  # central difference

# Test with f(x) = x^3
f    = lambda x: x**3
f_dx = lambda x: 3*x**2  # analytical derivative

x = 2.0
numerical = numerical_derivative(f, x)
analytical = f_dx(x)

print(f"x = {x}")
print(f"Numerical derivative:  {numerical:.6f}")   # ~12.000000
print(f"Analytical derivative: {analytical:.6f}")  # 12.000000

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)  # this is the elegant sigmoid derivative

x_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"\nx:          {x_vals}")
print(f"σ(x):       {sigmoid(x_vals).round(4)}")
print(f"σ'(x):      {sigmoid_deriv(x_vals).round(4)}")
# Peak at x=0, where σ(0)=0.5, σ'(0)=0.25
```

---

## 2.2 Partial Derivatives

When a function has multiple inputs, the partial derivative with respect to one variable treats all others as constants.

$$\frac{\partial f}{\partial x_i}\ \text{— how much does}\ f\ \text{change if we nudge only}\ x_i?$$

**Example:** $f(x, y) = x^2 y + 3xy^2$

$$\frac{\partial f}{\partial x} = 2xy + 3y^2 \qquad \frac{\partial f}{\partial y} = x^2 + 6xy$$

```python
import numpy as np

def f(x, y):
    return x**2 * y + 3 * x * y**2

# Analytical partial derivatives
def df_dx(x, y): return 2*x*y + 3*y**2
def df_dy(x, y): return x**2 + 6*x*y

# Numerical verification
def partial_x(f, x, y, h=1e-5):
    return (f(x+h, y) - f(x-h, y)) / (2*h)

def partial_y(f, x, y, h=1e-5):
    return (f(x, y+h) - f(x, y-h)) / (2*h)

x, y = 2.0, 3.0

print(f"f({x},{y}) = {f(x,y)}")    # 4*3 + 3*2*9 = 12 + 54 = 66

print(f"\n∂f/∂x analytical: {df_dx(x,y)}")        # 2*2*3 + 3*9 = 12+27=39
print(f"∂f/∂x numerical:  {partial_x(f,x,y):.6f}")  # ~39.0

print(f"\n∂f/∂y analytical: {df_dy(x,y)}")        # 4 + 6*2*3 = 4+36=40
print(f"∂f/∂y numerical:  {partial_y(f,x,y):.6f}")  # ~40.0
```

**ML connection:** In a neural network loss $\mathcal{L}(w_1, w_2, \ldots, w_n)$, we compute partial derivatives with respect to every weight. Each $\frac{\partial \mathcal{L}}{\partial w_i}$ tells us how much that specific weight is hurting performance.

---

## 2.3 Gradient — The Direction of Steepest Ascent

The gradient packages all partial derivatives into one vector:

$$\nabla_\mathbf{x} f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

The gradient **points toward steepest ascent**. Therefore $-\nabla f$ points toward steepest descent — the direction to move to reduce the loss.

```python
import numpy as np

# Bowl-shaped loss function: L(w1, w2) = w1^2 + w2^2
def loss(w):
    return w[0]**2 + w[1]**2

def gradient(w):
    # ∂L/∂w1 = 2*w1,  ∂L/∂w2 = 2*w2
    return np.array([2*w[0], 2*w[1]])

w = np.array([3.0, 4.0])
g = gradient(w)

print(f"Current weights: {w}")           # [3. 4.]
print(f"Loss:            {loss(w)}")     # 25.0
print(f"Gradient:        {g}")           # [6. 8.] — points AWAY from minimum
print(f"Gradient direction: uphill → moving -gradient takes us toward minimum")

# Gradient magnitude tells steepness
print(f"Gradient norm:   {np.linalg.norm(g):.2f}")  # 10.0 — steep!
```

---

## 2.4 Chain Rule

For composite function $f(g(x))$:
$$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$

For multi-variable composites (neural networks):
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### Why This is the Core of Deep Learning

A neural network is a nested function:
$$\mathcal{L}(\mathbf{w}) = \text{loss}(\underbrace{\sigma(\underbrace{W^{(2)} \cdot \sigma(\underbrace{W^{(1)} \mathbf{x}}_{z^{(1)}}) + b^{(2)}}_{z^{(2)}})}_{output})$$

Backpropagation just applies the chain rule systematically, layer by layer, from the loss back to the first layer.

```python
import numpy as np

# Manual backprop through 2-layer network (single sample)
# y_hat = sigmoid(w2 * sigmoid(w1 * x + b1) + b2)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward pass
x  = 2.0
w1 = 0.5;  b1 = -1.0
w2 = 1.5;  b2 = 0.0
y  = 1.0    # true label

z1    = w1 * x + b1          # pre-activation layer 1
a1    = sigmoid(z1)           # activation layer 1
z2    = w2 * a1 + b2          # pre-activation layer 2
y_hat = sigmoid(z2)           # final prediction

loss = -( y * np.log(y_hat) + (1-y) * np.log(1 - y_hat) )  # binary cross-entropy
print(f"z1={z1:.4f}, a1={a1:.4f}, z2={z2:.4f}, y_hat={y_hat:.4f}")
print(f"Loss: {loss:.4f}")

# Backward pass — chain rule
# dL/dy_hat
dL_dyhat = -(y / y_hat) + (1-y) / (1 - y_hat)

# dL/dz2 = dL/dy_hat * sigmoid'(z2)
dyhat_dz2 = y_hat * (1 - y_hat)
dL_dz2    = dL_dyhat * dyhat_dz2

# dL/dw2 = dL/dz2 * dz2/dw2 = dL/dz2 * a1
dL_dw2 = dL_dz2 * a1
print(f"\nGradient for w2: {dL_dw2:.6f}")

# dL/da1 = dL/dz2 * dz2/da1 = dL/dz2 * w2
dL_da1 = dL_dz2 * w2

# dL/dz1 = dL/da1 * sigmoid'(z1)
da1_dz1 = a1 * (1 - a1)
dL_dz1  = dL_da1 * da1_dz1

# dL/dw1 = dL/dz1 * dz1/dw1 = dL/dz1 * x
dL_dw1 = dL_dz1 * x
print(f"Gradient for w1: {dL_dw1:.6f}")

# Weight update (one gradient descent step)
lr = 0.1
w1 -= lr * dL_dw1
w2 -= lr * dL_dw2
print(f"\nUpdated w1: {w1:.4f}, w2: {w2:.4f}")
```

---

## 2.5 Gradient Descent

The fundamental optimization algorithm in all of ML:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_\mathbf{w} \mathcal{L}(\mathbf{w}_t)$$

where $\eta$ (eta) is the **learning rate** — how big a step to take.

### Types of Gradient Descent

| Type | Batch size | Update frequency | When to use |
|------|-----------|-----------------|-------------|
| Batch GD | Full dataset | Once per epoch | Small datasets, exact gradients |
| SGD | 1 sample | Every sample | Online learning |
| Mini-batch SGD | 32–512 | Every mini-batch | Deep learning (default) |

```python
import numpy as np

rng = np.random.default_rng(42)

# Generate synthetic data: y = 2x + 1 + noise
n = 100
X = rng.uniform(0, 10, n)
y = 2 * X + 1 + rng.normal(0, 1, n)  # true slope=2, intercept=1

# ── Gradient Descent from Scratch ────────────────────────────
w = 0.0   # slope (weight)
b = 0.0   # intercept (bias)
lr = 0.001  # learning rate
epochs = 500

loss_history = []

for epoch in range(epochs):
    # Forward pass: compute predictions
    y_pred = w * X + b

    # Compute MSE loss
    errors = y_pred - y
    loss   = (errors**2).mean()
    loss_history.append(loss)

    # Compute gradients
    # dL/dw = (2/n) * sum((y_pred - y) * x)
    # dL/db = (2/n) * sum(y_pred - y)
    dL_dw = (2/n) * (errors * X).sum()
    dL_db = (2/n) * errors.sum()

    # Update weights (move against gradient)
    w -= lr * dL_dw
    b -= lr * dL_db

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

print(f"\nFinal: w={w:.4f} (true=2), b={b:.4f} (true=1)")
print(f"Loss reduced from {loss_history[0]:.2f} to {loss_history[-1]:.4f}")
```

**Output walkthrough:**
- `errors = y_pred - y` — how wrong each prediction is
- `dL_dw = (2/n) * (errors * X).sum()` — gradient: each error, weighted by its x value (because w multiplies x)
- `w -= lr * dL_dw` — move slope in the direction that reduces loss
- After 500 steps, w ≈ 2.0 and b ≈ 1.0 — discovered the true relationship!

### Learning Rate Effects

```python
# Too small LR: converges slowly
# Too large LR: diverges
# Right LR: fast convergence

def train_linear(lr, epochs=200):
    w, b = 0.0, 0.0
    for _ in range(epochs):
        y_pred = w * X + b
        errors = y_pred - y
        w -= lr * (2/n) * (errors * X).sum()
        b -= lr * (2/n) * errors.sum()
    return w, b, ((w * X + b - y)**2).mean()

for lr in [0.0001, 0.001, 0.01, 0.1]:
    w, b, loss = train_linear(lr)
    print(f"lr={lr:.4f}: w={w:.3f}, b={b:.3f}, loss={loss:.4f}")
```

---

# PART 3 — Probability & Statistics

Probability is the mathematical framework for uncertainty. Every prediction a neural network makes is fundamentally a probability distribution.

---

## 3.1 Foundations

### Sample Space and Events

- **Sample space** $\Omega$: all possible outcomes
- **Event** $A \subseteq \Omega$: a subset of outcomes
- **Probability axioms:** $P(A) \geq 0$, $P(\Omega) = 1$, $P(A \cup B) = P(A) + P(B)$ for disjoint $A, B$

```python
import numpy as np

rng = np.random.default_rng(42)

# Simulating probabilities empirically
n_trials = 100_000

# Coin flip: P(heads) = 0.5
flips   = rng.integers(0, 2, n_trials)
p_heads = flips.mean()
print(f"Empirical P(heads): {p_heads:.4f}")  # ~0.5000

# Biased die: faces 1-6, face 6 appears with P=0.5, others share P=0.1
faces = rng.choice([1,2,3,4,5,6], p=[0.1,0.1,0.1,0.1,0.1,0.5], size=n_trials)
for face in range(1, 7):
    p_face = (faces == face).mean()
    print(f"  P(die={face}): {p_face:.4f}")
```

### Conditional Probability

$$P(A | B) = \frac{P(A \cap B)}{P(B)}$$

**Reading:** "Probability of A, given we already know B happened."

```python
# Example: students who study > 4 hours AND pass
rng = np.random.default_rng(42)
n   = 10_000

# Simulate: study_hours uniform [0, 8], pass if study > 3 + noise
study_hours = rng.uniform(0, 8, n)
passed      = (study_hours + rng.normal(0, 1, n)) > 3.5

# P(pass)
p_pass = passed.mean()

# P(study > 4)
studies_a_lot = study_hours > 4
p_studies     = studies_a_lot.mean()

# P(pass AND study > 4)
p_both = (passed & studies_a_lot).mean()

# P(pass | study > 4)
p_pass_given_study = p_both / p_studies

print(f"P(pass):                    {p_pass:.4f}")
print(f"P(study > 4):               {p_studies:.4f}")
print(f"P(pass | study > 4):        {p_pass_given_study:.4f}")
# Much higher — studying helps!
```

---

## 3.2 Bayes' Theorem

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

In ML terms (updating belief about model parameters $\theta$ after seeing data $D$):

$$\underbrace{P(\theta|D)}_{\text{posterior}} = \frac{\underbrace{P(D|\theta)}_{\text{likelihood}} \cdot \underbrace{P(\theta)}_{\text{prior}}}{\underbrace{P(D)}_{\text{evidence}}}$$

| Term | Meaning | Example |
|------|---------|---------|
| Prior $P(\theta)$ | Belief before data | "Weights are probably near zero" |
| Likelihood $P(D|\theta)$ | How well $\theta$ explains data | Loss function |
| Posterior $P(\theta|D)$ | Updated belief after data | Trained model weights |
| Evidence $P(D)$ | Normalization constant | Often intractable |

```python
# Naive Bayes spam classifier
# P(spam | words) ∝ P(words | spam) * P(spam)

spam_examples = [
    "buy cheap meds now",
    "free prize claim immediately",
    "win lottery money fast",
]
ham_examples = [
    "meeting tomorrow at 3pm",
    "please review the report",
    "see you at the conference",
]

all_words_spam = " ".join(spam_examples).split()
all_words_ham  = " ".join(ham_examples).split()

# Vocabulary
vocab = set(all_words_spam + all_words_ham)

# Prior: P(spam) = 0.3 (30% of emails are spam)
p_spam = 0.3
p_ham  = 0.7

# Likelihood: P(word | class) with Laplace smoothing
def word_prob(word, word_list, vocab_size, alpha=1):
    count = word_list.count(word) + alpha
    total = len(word_list) + alpha * vocab_size
    return count / total

def classify(email, vocab=vocab):
    words = email.split()
    # Log probabilities to avoid underflow
    log_p_spam = np.log(p_spam)
    log_p_ham  = np.log(p_ham)

    for word in words:
        log_p_spam += np.log(word_prob(word, all_words_spam, len(vocab)))
        log_p_ham  += np.log(word_prob(word, all_words_ham,  len(vocab)))

    return "SPAM" if log_p_spam > log_p_ham else "HAM"

test_emails = [
    "buy cheap prize now",
    "meeting at conference tomorrow",
    "free money win fast",
]

for email in test_emails:
    print(f"'{email}' → {classify(email)}")
```

---

## 3.3 Random Variables & Key Distributions

### Expectation and Variance

$$\mathbb{E}[X] = \sum_x x \cdot P(X=x) \qquad \text{(discrete)}$$

$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

```python
import numpy as np

rng = np.random.default_rng(42)

# Bernoulli(p): 1 with prob p, 0 with prob 1-p
p = 0.7
samples = rng.binomial(1, p, size=10_000)

print(f"Bernoulli(0.7):")
print(f"  Theoretical mean:     {p:.4f}")
print(f"  Empirical mean:       {samples.mean():.4f}")
print(f"  Theoretical variance: {p*(1-p):.4f}")
print(f"  Empirical variance:   {samples.var():.4f}")

# Gaussian: N(μ, σ²)
mu, sigma = 70, 15  # exam scores: mean=70, std=15
scores = rng.normal(mu, sigma, 10_000)

print(f"\nGaussian(70, 15²):")
print(f"  Mean:    {scores.mean():.2f}")
print(f"  Std:     {scores.std():.2f}")
print(f"  Within 1σ: {((scores > mu-sigma) & (scores < mu+sigma)).mean():.1%}")  # ~68%
print(f"  Within 2σ: {((scores > mu-2*sigma) & (scores < mu+2*sigma)).mean():.1%}")  # ~95%
print(f"  Within 3σ: {((scores > mu-3*sigma) & (scores < mu+3*sigma)).mean():.1%}")  # ~99.7%
```

### Gaussian PDF

$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

```python
def gaussian_pdf(x, mu, sigma):
    # Normalization constant
    norm  = 1 / (sigma * np.sqrt(2 * np.pi))
    # Exponent
    exponent = np.exp(-0.5 * ((x - mu) / sigma)**2)
    return norm * exponent

x_vals = np.linspace(30, 110, 9)
probs  = gaussian_pdf(x_vals, mu=70, sigma=15)

print("Score → P(score):")
for x, p in zip(x_vals, probs):
    bar = "█" * int(p * 300)
    print(f"  {x:5.1f}: {p:.5f} {bar}")
```

---

## 3.4 Maximum Likelihood Estimation (MLE)

**Core idea:** Find parameters $\theta$ that make the observed data most probable.

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^N p(x^{(i)} ; \theta) = \arg\max_\theta \sum_{i=1}^N \log p(x^{(i)}; \theta)$$

(log converts product → sum, making it numerically stable and easier to differentiate)

```python
import numpy as np

rng = np.random.default_rng(42)

# Data: exam scores assumed to follow N(μ, σ²)
true_mu, true_sigma = 72.0, 12.0
data = rng.normal(true_mu, true_sigma, 1000)

# MLE for Gaussian:
# d/dμ log L = 0 → μ_MLE = sample mean
# d/dσ log L = 0 → σ_MLE = sample std (biased)

mu_mle    = data.mean()
sigma_mle = data.std(ddof=0)  # ddof=0 → biased MLE

print(f"True parameters:  μ={true_mu}, σ={true_sigma}")
print(f"MLE estimates:    μ={mu_mle:.4f}, σ={sigma_mle:.4f}")

# The log-likelihood at the MLE solution
def log_likelihood(data, mu, sigma):
    n   = len(data)
    ll  = -n/2 * np.log(2 * np.pi * sigma**2)
    ll -= 1/(2 * sigma**2) * ((data - mu)**2).sum()
    return ll

ll_true = log_likelihood(data, true_mu, true_sigma)
ll_mle  = log_likelihood(data, mu_mle, sigma_mle)
print(f"\nLog-likelihood (true params): {ll_true:.2f}")
print(f"Log-likelihood (MLE):         {ll_mle:.2f}")
print(f"MLE is always >= true params: {ll_mle >= ll_true}")
```

---

## 3.5 Cross-Entropy and KL Divergence

### Shannon Entropy

$$H(P) = -\sum_x P(x) \log P(x)$$

Entropy measures uncertainty. A uniform distribution has maximum entropy; a peaked distribution has low entropy.

### Cross-Entropy Loss (Classification)

$$\mathcal{L}_{\text{CE}} = -\sum_k y_k \log \hat{y}_k$$

For one-hot labels (only one class is correct): $\mathcal{L}_{\text{CE}} = -\log \hat{y}_{y^*}$

```python
import numpy as np

def cross_entropy(y_true, y_pred, eps=1e-9):
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred))

# Binary case: cat vs not-cat
y_true = np.array([1, 0])   # it IS a cat

# Confident correct prediction
y_pred_good = np.array([0.95, 0.05])
print(f"Confident correct:  loss={cross_entropy(y_true, y_pred_good):.4f}")  # small

# Uncertain prediction
y_pred_meh  = np.array([0.55, 0.45])
print(f"Uncertain:          loss={cross_entropy(y_true, y_pred_meh):.4f}")   # medium

# Confident WRONG prediction
y_pred_bad  = np.array([0.05, 0.95])
print(f"Confident wrong:    loss={cross_entropy(y_true, y_pred_bad):.4f}")   # large!

# Multi-class: dog/cat/bird
y_true_mc  = np.array([0, 1, 0])   # it's a cat (index 1)
y_pred_mc  = np.array([0.1, 0.8, 0.1])   # 80% cat
print(f"\nMulticlass correct: loss={cross_entropy(y_true_mc, y_pred_mc):.4f}")

# Softmax for multi-class outputs
def softmax(z):
    e = np.exp(z - z.max())  # subtract max for numerical stability
    return e / e.sum()

logits = np.array([1.5, 3.2, 0.8])   # raw model outputs (not probabilities)
probs  = softmax(logits)
print(f"\nLogits: {logits}")
print(f"Probs:  {probs.round(4)}")    # sums to 1
print(f"Sum:    {probs.sum():.6f}")   # 1.000000
```

### KL Divergence

$$D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} \geq 0$$

Measures how different distribution $Q$ is from reference $P$. **Not symmetric:** $D_{\text{KL}}(P\|Q) \neq D_{\text{KL}}(Q\|P)$.

```python
def kl_divergence(P, Q, eps=1e-9):
    P = np.clip(P, eps, 1)
    Q = np.clip(Q, eps, 1)
    return np.sum(P * np.log(P / Q))

# P = true distribution, Q = model's distribution
P = np.array([0.1, 0.4, 0.5])   # true class distribution
Q = np.array([0.2, 0.3, 0.5])   # model's prediction

print(f"KL(P||Q): {kl_divergence(P, Q):.6f}")   # > 0
print(f"KL(Q||P): {kl_divergence(Q, P):.6f}")   # different!

# When Q = P, KL = 0
print(f"KL(P||P): {kl_divergence(P, P):.6f}")   # 0.000000

# Relationship: H(P,Q) = H(P) + KL(P||Q)
def entropy(p, eps=1e-9):
    p = np.clip(p, eps, 1)
    return -np.sum(p * np.log(p))

H_P = entropy(P)
CE  = cross_entropy(P, Q)
KL  = kl_divergence(P, Q)
print(f"\nH(P) + KL(P||Q) = {H_P + KL:.6f}")
print(f"H(P, Q)          = {CE:.6f}")
print(f"Equal: {np.isclose(H_P + KL, CE)}")  # True
```

**ML connection:** Training minimizes cross-entropy, which is equivalent to minimizing KL divergence between the true label distribution and model predictions. VAEs minimize KL divergence between the learned latent distribution and a standard Gaussian prior.

---

# PART 4 — Interview Q&A

## Q1: Why must matrix multiplication dimensions match?

Matrix multiply computes dot products between rows of $A$ and columns of $B$. The rows of $A$ have $k$ elements; the columns of $B$ must also have $k$ elements to compute the dot product.

## Q2: What is the difference between L1 and L2 regularization?

**L1 (Lasso):** Penalty $\lambda \|\mathbf{w}\|_1$. Gradient is constant ($\pm\lambda$) regardless of weight magnitude, so small weights can be driven exactly to zero. Creates sparse models.

**L2 (Ridge):** Penalty $\lambda \|\mathbf{w}\|_2^2$. Gradient is $2\lambda w_i$, proportional to weight size. Shrinks all weights but never to exactly zero. Bayesian interpretation: Gaussian prior.

## Q3: Why do we use log-likelihood instead of likelihood?

Products of many small probabilities ($\prod p_i$ where each $p_i < 1$) quickly underflow floating-point precision. Logarithm converts products to sums ($\sum \log p_i$), stable and differentiable everywhere. Maximizing log-likelihood is equivalent to maximizing likelihood since log is monotone increasing.

## Q4: Explain backpropagation in one sentence.

Backprop applies the chain rule layer-by-layer from loss to inputs, computing each parameter's gradient as the product of all downstream derivatives.

## Q5: What is the gradient of cross-entropy loss with respect to the softmax input?

$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_k} = \hat{y}_k - y_k$$

The simplest gradient in deep learning: prediction minus label. This elegant form emerges from combining the softmax derivative and cross-entropy derivative.

## Q6: Why does PCA use eigendecomposition?

PCA finds axes of maximum variance. Variance in direction $\mathbf{v}$ is $\mathbf{v}^T \Sigma \mathbf{v}$ (quadratic form). Maximizing this subject to $\|\mathbf{v}\|=1$ is exactly the eigenvector problem for covariance matrix $\Sigma$.

## Q7: What does SVD tell you that eigendecomposition cannot?

Eigendecomposition requires square matrices. SVD works for any shape — a 10,000×768 embedding matrix can be SVD-decomposed. The singular values reveal the effective rank (how many meaningful dimensions data actually occupies).

---

# PART 5 — Cheat Sheet

| Formula | Name | ML Role |
|---------|------|---------|
| $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$ | Dot product | Attention, similarity, layer computation |
| $\text{cosim}(\mathbf{u},\mathbf{v}) = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ | Cosine similarity | Embedding search, RAG retrieval |
| $A = U\Sigma V^T$ | SVD | Compression, recommendation, LoRA |
| $A\mathbf{v} = \lambda\mathbf{v}$ | Eigendecomposition | PCA, spectral clustering |
| $\mathbf{w} \leftarrow \mathbf{w} - \eta\nabla\mathcal{L}$ | Gradient descent | All neural network training |
| $\frac{dL}{dw} = \frac{dL}{d\hat{y}} \cdot \frac{d\hat{y}}{dz} \cdot \frac{dz}{dw}$ | Chain rule | Backpropagation |
| $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$ | Bayes theorem | Bayesian ML, spam filter |
| $\hat{\theta} = \arg\max \sum \log p(x|\theta)$ | MLE | Loss function derivation |
| $H(P,Q) = -\sum P(x)\log Q(x)$ | Cross-entropy | Classification loss |
| $D_{\text{KL}}(P\|Q) = \sum P\log\frac{P}{Q}$ | KL divergence | VAE, RLHF, distribution matching |

---

# MINI-PROJECT — Grade Predictor from Scratch

**What you will build:** A linear regression model trained with gradient descent that predicts a student's final exam score from study hours, sleep hours, and attendance percentage. Every line of code uses math from this module.

**Learning goals:**
- Vectors = student feature profiles
- Matrix operations = batch prediction
- Gradient descent = learning from errors
- Cross-entropy / MSE = measuring how wrong we are

---

## Step 1 — Generate Synthetic Dataset

```python
import numpy as np

rng = np.random.default_rng(42)
n   = 200  # 200 students

# Features: study hours (0-8), sleep hours (4-9), attendance (50-100)
study      = rng.uniform(0, 8, n)
sleep      = rng.uniform(4, 9, n)
attendance = rng.uniform(50, 100, n)

# True relationship: score = 5*study + 2*sleep + 0.3*attendance + noise
true_weights = np.array([5.0, 2.0, 0.3])
bias_true    = 10.0

X = np.column_stack([study, sleep, attendance])  # shape (200, 3)
y = X @ true_weights + bias_true + rng.normal(0, 3, n)  # add noise

# Print what we're working with
print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"\nFirst 3 students:")
print(f"  {'Study':>8} {'Sleep':>8} {'Attend':>8} {'Score':>8}")
for i in range(3):
    print(f"  {X[i,0]:8.1f} {X[i,1]:8.1f} {X[i,2]:8.1f} {y[i]:8.1f}")

# Statistics
print(f"\nScore stats: mean={y.mean():.1f}, std={y.std():.1f}, "
      f"min={y.min():.1f}, max={y.max():.1f}")
```

---

## Step 2 — Normalize Features

Neural networks and gradient descent converge faster when features are on similar scales. Study hours (0–8) and attendance (50–100) are very different scales.

```python
# Standardize: (x - mean) / std  →  mean=0, std=1
X_mean = X.mean(axis=0)  # shape (3,) — one mean per feature
X_std  = X.std(axis=0)

X_norm = (X - X_mean) / X_std

print("Before normalization:")
print(f"  Study hours:  mean={X[:,0].mean():.2f}, std={X[:,0].std():.2f}")
print(f"  Sleep hours:  mean={X[:,1].mean():.2f}, std={X[:,1].std():.2f}")
print(f"  Attendance:   mean={X[:,2].mean():.2f}, std={X[:,2].std():.2f}")

print("\nAfter normalization:")
print(f"  Study hours:  mean={X_norm[:,0].mean():.4f}, std={X_norm[:,0].std():.4f}")
print(f"  Sleep hours:  mean={X_norm[:,1].mean():.4f}, std={X_norm[:,1].std():.4f}")
print(f"  Attendance:   mean={X_norm[:,2].mean():.4f}, std={X_norm[:,2].std():.4f}")
# All means ~0, all stds ~1
```

---

## Step 3 — Train / Validation Split

Never evaluate model performance on training data — it can memorize noise.

```python
# 80% training, 20% validation
split    = int(0.8 * n)
idx      = rng.permutation(n)  # shuffle indices

train_idx = idx[:split]
val_idx   = idx[split:]

X_train, y_train = X_norm[train_idx], y[train_idx]
X_val,   y_val   = X_norm[val_idx],   y[val_idx]

print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
```

---

## Step 4 — Linear Regression with Gradient Descent

```python
# Initialize weights to zero
w = np.zeros(3)   # one weight per feature
b = 0.0           # bias term

lr     = 0.01
epochs = 1000

train_losses = []
val_losses   = []

for epoch in range(epochs):
    # ── FORWARD PASS ──────────────────────────────────────
    # y_pred = X @ w + b
    # X_train: (160, 3), w: (3,) → y_pred: (160,)
    y_pred_train = X_train @ w + b

    # ── COMPUTE MSE LOSS ──────────────────────────────────
    errors = y_pred_train - y_train       # shape (160,)
    mse    = (errors**2).mean()           # scalar
    train_losses.append(mse)

    # ── COMPUTE GRADIENTS ─────────────────────────────────
    # dMSE/dw = (2/n) * X^T @ errors
    # dMSE/db = (2/n) * sum(errors)
    n_train = len(y_train)
    dL_dw   = (2 / n_train) * (X_train.T @ errors)  # shape (3,)
    dL_db   = (2 / n_train) * errors.sum()            # scalar

    # ── UPDATE WEIGHTS ────────────────────────────────────
    w -= lr * dL_dw
    b -= lr * dL_db

    # ── VALIDATION LOSS ───────────────────────────────────
    y_pred_val = X_val @ w + b
    val_mse    = ((y_pred_val - y_val)**2).mean()
    val_losses.append(val_mse)

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}: train_loss={mse:.4f}, val_loss={val_mse:.4f}")

print(f"\nFinal weights: {w.round(4)}")
print(f"Final bias:    {b:.4f}")
```

---

## Step 5 — Evaluate the Model

```python
# Predictions on validation set
y_pred_val = X_val @ w + b

# Mean Absolute Error
mae = np.abs(y_pred_val - y_val).mean()

# Root Mean Squared Error
rmse = np.sqrt(((y_pred_val - y_val)**2).mean())

# R² score (coefficient of determination)
ss_res = ((y_val - y_pred_val)**2).sum()
ss_tot = ((y_val - y_val.mean())**2).sum()
r2     = 1 - ss_res / ss_tot

print(f"Validation MAE:  {mae:.2f} points")
print(f"Validation RMSE: {rmse:.2f} points")
print(f"R² score:        {r2:.4f}  (1.0 = perfect)")

# Look at a few predictions vs actual
print(f"\n{'Predicted':>12} {'Actual':>10} {'Error':>8}")
for pred, actual in zip(y_pred_val[:5], y_val[:5]):
    print(f"{pred:12.1f} {actual:10.1f} {actual - pred:8.1f}")
```

---

## Step 6 — Understand What the Model Learned

```python
# The weights are in normalized space; convert back to interpret
# w_i_original = w_i / X_std_i
w_original = w / X_std

print("Feature importance (in original units):")
feature_names = ["Study hours", "Sleep hours", "Attendance %"]
for name, weight in zip(feature_names, w_original):
    print(f"  {name:<15}: {weight:+.4f} points per unit increase")

# Compare to ground truth
print(f"\nTrue weights: {true_weights}")
print(f"Learned:      {w_original.round(4)}")
```

---

## Step 7 — Predict for a New Student

```python
# New student: studies 6 hours, sleeps 7 hours, 85% attendance
new_student = np.array([[6.0, 7.0, 85.0]])

# Normalize using TRAINING statistics (never use test/new data to fit normalizer)
new_norm = (new_student - X_mean) / X_std

# Predict
predicted_score = new_norm @ w + b
print(f"Student profile: study=6h, sleep=7h, attendance=85%")
print(f"Predicted score: {predicted_score[0]:.1f}")

# Compute what the true model would predict (without noise)
true_prediction = new_student[0] @ true_weights + bias_true
print(f"True model prediction: {true_prediction:.1f}")
print(f"Our model error: {abs(predicted_score[0] - true_prediction):.1f} points")
```

---

## Step 8 — Loss Curve Analysis

```python
# Print training dynamics
print("Training progress summary:")
milestones = [0, 100, 200, 500, 999]
print(f"  {'Epoch':>8} {'Train Loss':>12} {'Val Loss':>12}")
for ep in milestones:
    print(f"  {ep:>8} {train_losses[ep]:>12.4f} {val_losses[ep]:>12.4f}")

# Convergence check: how much improvement in last 100 epochs?
improvement = train_losses[-101] - train_losses[-1]
print(f"\nLoss improvement in last 100 epochs: {improvement:.6f}")
print(f"Converged: {improvement < 0.01}")

# Check for overfitting: val_loss should be close to train_loss
gap = val_losses[-1] - train_losses[-1]
print(f"Train/Val gap: {gap:.4f} (small = no overfitting)")
```

---

## What This Project Demonstrated

| Concept | Where it appeared |
|---------|------------------|
| Vectors | Student profiles: `np.array([6, 7, 85])` |
| Matrix multiply | Batch prediction: `X_train @ w` |
| L2 norm | RMSE: `sqrt(mean(errors²))` |
| Gradient descent | Weight updates: `w -= lr * dL_dw` |
| Partial derivatives | `dL/dw = (2/n) * X^T @ errors` |
| Chain rule | The gradient formula itself |
| MLE | MSE loss = MLE under Gaussian noise |
| Normalization | Zero-mean unit-variance features |

Every concept from this module appeared in one coherent project. This is exactly how senior engineers think: math → code → model → evaluation.

---

## Complete Project Code (All Steps Together)

```python
import numpy as np

def main():
    rng = np.random.default_rng(42)
    n   = 200

    # Data
    study      = rng.uniform(0, 8, n)
    sleep      = rng.uniform(4, 9, n)
    attendance = rng.uniform(50, 100, n)
    X = np.column_stack([study, sleep, attendance])
    y = X @ np.array([5.0, 2.0, 0.3]) + 10.0 + rng.normal(0, 3, n)

    # Normalize
    X_mean = X.mean(axis=0); X_std = X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    # Split
    idx = rng.permutation(n)
    X_train, y_train = X_norm[idx[:160]], y[idx[:160]]
    X_val,   y_val   = X_norm[idx[160:]], y[idx[160:]]

    # Train
    w, b = np.zeros(3), 0.0
    for _ in range(1000):
        errors = X_train @ w + b - y_train
        w -= 0.01 * (2/160) * (X_train.T @ errors)
        b -= 0.01 * (2/160) * errors.sum()

    # Evaluate
    y_pred = X_val @ w + b
    rmse   = np.sqrt(((y_pred - y_val)**2).mean())
    r2     = 1 - ((y_val - y_pred)**2).sum() / ((y_val - y_val.mean())**2).sum()
    print(f"RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # Predict new student
    new = (np.array([[6.0, 7.0, 85.0]]) - X_mean) / X_std
    print(f"New student predicted score: {(new @ w + b)[0]:.1f}")

if __name__ == "__main__":
    main()
```

---

*Next: [Module 02 — ML Basics to Advanced](02-ml-basics.md)*
