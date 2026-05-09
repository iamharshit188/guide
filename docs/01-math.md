# Module 01 — Math for ML

> **Runnable code:** `src/01-math/`
> ```bash
> cd src/01-math
> python vectors.py
> python matrix_ops.py
> python calculus_demo.py
> python probability.py
> ```

---

## 1. Linear Algebra

### 1.1 Scalars, Vectors, Matrices, Tensors

| Object | Notation | Shape | Example |
|--------|----------|-------|---------|
| Scalar | $x \in \mathbb{R}$ | `()` | `3.14` |
| Vector | $\mathbf{v} \in \mathbb{R}^n$ | `(n,)` | `[1, 2, 3]` |
| Matrix | $A \in \mathbb{R}^{m \times n}$ | `(m, n)` | `[[1,2],[3,4]]` |
| Tensor | $\mathcal{T} \in \mathbb{R}^{d_1 \times \cdots \times d_k}$ | `(d1,...,dk)` | batch of images |

### 1.2 Vector Operations

**Addition:**
$$\mathbf{u} + \mathbf{v} = [u_1 + v_1,\; u_2 + v_2,\; \ldots,\; u_n + v_n]$$

**Scalar multiplication:**
$$c\mathbf{v} = [cv_1, cv_2, \ldots, cv_n]$$

**Dot product (inner product):**
$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = \mathbf{u}^T \mathbf{v}$$

Geometric interpretation: $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$

**Why it matters:** Dot products underlie every weighted sum in neural networks, attention scores, and similarity computations.

### 1.3 Norms

The **p-norm** of a vector:

$$\|\mathbf{v}\|_p = \left(\sum_{i=1}^n |v_i|^p\right)^{1/p}$$

| Norm | Formula | Use case |
|------|---------|----------|
| L1 (Manhattan) | $\sum_i \|v_i\|$ | Lasso regularization (sparsity) |
| L2 (Euclidean) | $\sqrt{\sum_i v_i^2}$ | Ridge regularization, distances |
| L∞ (Max) | $\max_i \|v_i\|$ | Adversarial robustness |

**Cosine similarity** (direction, not magnitude):
$$\text{cos\_sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \in [-1, 1]$$

Used in: embedding search, recommendation systems, RAG retrieval.

**Vector projection** of $\mathbf{u}$ onto $\mathbf{v}$:
$$\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\mathbf{v} \cdot \mathbf{v}} \mathbf{v}$$

### 1.4 Matrix Operations

**Matrix multiplication** $C = AB$ where $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$:

$$C_{ij} = \sum_{l=1}^k A_{il} B_{lj}$$

Dimensions: $(m \times k)(k \times n) \to (m \times n)$. Inner dimensions must match.

**Properties:**
- Associative: $(AB)C = A(BC)$
- Distributive: $A(B+C) = AB + AC$
- **NOT commutative:** $AB \neq BA$ in general
- Transpose of product: $(AB)^T = B^T A^T$

**Trace:** Sum of diagonal elements. $\text{tr}(A) = \sum_i A_{ii}$

**Determinant** (2×2): $\det\begin{pmatrix}a & b \\ c & d\end{pmatrix} = ad - bc$

A matrix is invertible iff $\det(A) \neq 0$.

### 1.5 Matrix Inverse

For square $A \in \mathbb{R}^{n \times n}$:
$$A A^{-1} = A^{-1} A = I$$

$(AB)^{-1} = B^{-1} A^{-1}$

**Moore-Penrose Pseudoinverse** (for non-square or singular matrices):
$$A^+ = (A^T A)^{-1} A^T \quad \text{(when } A^T A \text{ is invertible)}$$

Used in least-squares: $\hat{\mathbf{w}} = A^+ \mathbf{b}$ solves overdetermined systems.

### 1.6 Eigenvalues & Eigenvectors

For matrix $A$, vector $\mathbf{v} \neq \mathbf{0}$, scalar $\lambda$:

$$A\mathbf{v} = \lambda \mathbf{v}$$

$\mathbf{v}$ is an **eigenvector**, $\lambda$ is the corresponding **eigenvalue**.

**Finding eigenvalues:** Solve the characteristic equation:
$$\det(A - \lambda I) = 0$$

**Eigendecomposition** (requires $n$ linearly independent eigenvectors):
$$A = V \Lambda V^{-1}$$

where $V$ = matrix of eigenvectors (columns), $\Lambda$ = diagonal matrix of eigenvalues.

**For symmetric matrices** ($A = A^T$, e.g., covariance matrices):
$$A = Q \Lambda Q^T, \quad Q^T Q = I \quad \text{(orthonormal eigenvectors)}$$

**ML applications:**
- **PCA:** eigenvectors of covariance matrix = principal components; eigenvalues = variance explained
- **PageRank:** dominant eigenvector of transition matrix
- **Stability analysis:** eigenvalues of Jacobian determine convergence

### 1.7 Singular Value Decomposition (SVD)

Any matrix $A \in \mathbb{R}^{m \times n}$ decomposes as:

$$A = U \Sigma V^T$$

| Matrix | Shape | Meaning |
|--------|-------|---------|
| $U$ | $m \times m$ | Left singular vectors (orthonormal) |
| $\Sigma$ | $m \times n$ | Diagonal — singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ |
| $V^T$ | $n \times n$ | Right singular vectors (orthonormal) |

**Relation to eigendecomp:**
$$A^T A = V \Sigma^T \Sigma V^T \quad \Rightarrow \quad \sigma_i = \sqrt{\lambda_i(A^T A)}$$

**Truncated SVD** (rank-$k$ approximation, optimal by Eckart-Young theorem):
$$A \approx U_k \Sigma_k V_k^T$$

**ML applications:**
- Dimensionality reduction (equivalent to PCA)
- Latent Semantic Analysis (LSA)
- Matrix factorization for recommender systems
- Numerical stability (solving least-squares)

> **Run:** `python src/01-math/matrix_ops.py` — demonstrates eigendecomp, SVD, pseudoinverse

---

## 2. Calculus for ML

### 2.1 Partial Derivatives

For $f: \mathbb{R}^n \to \mathbb{R}$, the partial derivative w.r.t. $x_i$:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(\ldots, x_i + h, \ldots) - f(\ldots, x_i, \ldots)}{h}$$

Treat all other variables as constants.

**Example:** $f(x, y) = x^2 y + 3xy^2$

$$\frac{\partial f}{\partial x} = 2xy + 3y^2, \quad \frac{\partial f}{\partial y} = x^2 + 6xy$$

### 2.2 Gradient

The gradient is the vector of all partial derivatives:

$$\nabla_{\mathbf{x}} f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Key property:** $\nabla f$ points in the direction of steepest ascent. $-\nabla f$ points toward steepest descent.

### 2.3 Chain Rule

For composite function $f(g(x))$:
$$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$

**Multivariate chain rule** — if $z = f(x, y)$ and $x = g(t)$, $y = h(t)$:
$$\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt}$$

**Why this matters:** Backpropagation IS the chain rule applied recursively through a computational graph. Every gradient computation in deep learning uses this.

### 2.4 Jacobian and Hessian

**Jacobian** — for $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:
$$J_{ij} = \frac{\partial f_i}{\partial x_j}, \quad J \in \mathbb{R}^{m \times n}$$

**Hessian** — for $f: \mathbb{R}^n \to \mathbb{R}$, matrix of second-order partials:
$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}, \quad H \in \mathbb{R}^{n \times n}$$

Hessian is symmetric. Its eigenvalues determine the curvature:
- All positive eigenvalues → local minimum (positive definite)
- All negative → local maximum
- Mixed signs → saddle point

### 2.5 Gradient Descent

**Objective:** Minimize $\mathcal{L}(\mathbf{w})$ over parameters $\mathbf{w}$.

**Update rule:**
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}_t)$$

where $\eta > 0$ is the **learning rate**.

**Variants:**

| Variant | Gradient estimated from | Update frequency |
|---------|------------------------|-----------------|
| Batch GD | Full dataset | Once per epoch |
| Mini-batch SGD | Subset (batch size $B$) | $N/B$ times per epoch |
| SGD | Single sample | $N$ times per epoch |

**Convergence guarantee:** For convex $\mathcal{L}$ with Lipschitz gradients, GD converges if $\eta < \frac{2}{L}$ where $L$ is the Lipschitz constant.

**Numerical gradient check** (for debugging):
$$\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + \epsilon \mathbf{e}_i) - f(\mathbf{x} - \epsilon \mathbf{e}_i)}{2\epsilon}$$

Use $\epsilon = 10^{-5}$. Compare with analytical gradient; relative error should be $< 10^{-4}$.

> **Run:** `python src/01-math/calculus_demo.py` — numerical vs analytical gradients, gradient descent on quadratic

---

## 3. Probability & Statistics

### 3.1 Foundations

**Sample space** $\Omega$: set of all outcomes.
**Event** $A \subseteq \Omega$. **Probability** $P: \text{events} \to [0,1]$.

**Kolmogorov axioms:**
1. $P(A) \geq 0$
2. $P(\Omega) = 1$
3. $P(A \cup B) = P(A) + P(B)$ if $A \cap B = \emptyset$

**Joint probability:** $P(A \cap B) = P(A,B)$

**Conditional probability:**
$$P(A \mid B) = \frac{P(A, B)}{P(B)}, \quad P(B) > 0$$

**Independence:** $A \perp B \iff P(A, B) = P(A)P(B)$

### 3.2 Bayes' Theorem

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

In ML terminology:
$$P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) \cdot P(\theta)}{P(\mathcal{D})}$$

| Term | Name | Meaning |
|------|------|---------|
| $P(\theta \mid \mathcal{D})$ | Posterior | Updated belief after seeing data |
| $P(\mathcal{D} \mid \theta)$ | Likelihood | How probable is data given parameters |
| $P(\theta)$ | Prior | Belief before seeing data |
| $P(\mathcal{D})$ | Evidence / Marginal | Normalizing constant |

**Law of total probability:**
$$P(B) = \sum_i P(B \mid A_i) P(A_i) \quad \text{(partition of } \Omega\text{)}$$

### 3.3 Random Variables & Distributions

**Expectation:**
$$\mathbb{E}[X] = \sum_x x \cdot P(X=x) \quad \text{(discrete)}, \quad \int_{-\infty}^{\infty} x \cdot p(x)\, dx \quad \text{(continuous)}$$

**Variance:**
$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**Covariance:**
$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$$

**Covariance matrix** for vector $\mathbf{x} \in \mathbb{R}^n$:
$$\Sigma_{ij} = \text{Cov}(X_i, X_j), \quad \Sigma \in \mathbb{R}^{n \times n}, \quad \Sigma \text{ is symmetric positive semidefinite}$$

### 3.4 Key Distributions

**Bernoulli** — single binary trial:
$$P(X=1) = p, \quad P(X=0) = 1-p, \quad \mathbb{E}[X] = p, \quad \text{Var}(X) = p(1-p)$$

**Categorical** — $K$ classes, $\sum_k p_k = 1$:
$$P(X=k) = p_k, \quad \mathbb{E}[X_k] = p_k$$

**Gaussian (Normal)**:
$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \quad \mathbb{E}[X]=\mu, \quad \text{Var}(X)=\sigma^2$$

**Multivariate Gaussian** $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$:
$$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

The exponent $(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})$ is the **Mahalanobis distance** squared.

### 3.5 Maximum Likelihood Estimation (MLE)

Given data $\mathcal{D} = \{x^{(1)}, \ldots, x^{(N)}\}$ i.i.d. from $p(x; \theta)$:

**Likelihood:**
$$\mathcal{L}(\theta) = \prod_{i=1}^N p(x^{(i)}; \theta)$$

**Log-likelihood** (numerically stable, same argmax):
$$\ell(\theta) = \sum_{i=1}^N \log p(x^{(i)}; \theta)$$

**MLE:** $\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta)$

**Example — Gaussian MLE:** Given $x^{(i)} \sim \mathcal{N}(\mu, \sigma^2)$:
$$\ell(\mu, \sigma^2) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i(x^{(i)} - \mu)^2$$

Setting $\frac{\partial \ell}{\partial \mu} = 0$ and $\frac{\partial \ell}{\partial \sigma^2} = 0$:
$$\hat{\mu} = \frac{1}{N}\sum_i x^{(i)}, \quad \hat{\sigma}^2 = \frac{1}{N}\sum_i(x^{(i)} - \hat{\mu})^2$$

### 3.6 Cross-Entropy & KL Divergence

**Shannon entropy** — expected information content:
$$H(P) = -\sum_x P(x) \log P(x) = \mathbb{E}_P[-\log P(x)]$$

Maximized by uniform distribution; zero when distribution is deterministic.

**Cross-entropy** of $Q$ relative to $P$:
$$H(P, Q) = -\sum_x P(x) \log Q(x) = \mathbb{E}_P[-\log Q(x)]$$

Used as loss in classification: $P$ = true labels (one-hot), $Q$ = model predictions (softmax).

$$\mathcal{L}_{\text{CE}} = -\sum_k y_k \log \hat{y}_k = -\log \hat{y}_{y^*}$$

(simplifies to $-\log$ of predicted probability for the correct class)

**KL Divergence** (not a distance — asymmetric):
$$D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = H(P, Q) - H(P)$$

$D_{\text{KL}}(P \| Q) \geq 0$; equals 0 iff $P = Q$ a.e. (Gibbs inequality)

**Relationship:** $H(P, Q) = H(P) + D_{\text{KL}}(P \| Q)$

When $H(P)$ is constant (fixed labels), minimizing cross-entropy = minimizing KL divergence.

> **Run:** `python src/01-math/probability.py` — MLE fitting, KL divergence, entropy calculations

---

## Summary: Why This Math Underlies Everything

| ML Concept | Math Foundation |
|-----------|----------------|
| Neural network forward pass | Matrix multiplication |
| Backpropagation | Chain rule + partial derivatives |
| PCA | Eigendecomposition of covariance |
| Attention mechanism | Dot product + softmax |
| Loss functions | Cross-entropy, KL divergence |
| Model training | Gradient descent |
| Embedding similarity | Cosine similarity, L2 norm |
| Bayesian ML | Bayes theorem |
| Generative models | Gaussian / categorical distributions |
| Regularization | L1/L2 norms |

---

*Next: [Module 02 — ML Basics to Advanced](02-ml-basics.md)*
