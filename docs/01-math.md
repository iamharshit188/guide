# Module 01 — Math for ML

> **Runnable code:** `src/01-math/`
>
> ```bash
> cd src/01-math
> python vectors.py
> python matrix_ops.py
> python calculus_demo.py
> python probability.py
> ```

---

# Prerequisites & Overview

**Prerequisites:** High school algebra, basic Python/NumPy. No prior ML knowledge needed.

**Estimated time:** 6–10 hours (reading + running all four scripts)

---

# Why This Module Matters

Machine Learning is fundamentally applied mathematics.

Every modern ML system — from recommendation engines and chatbots to image classifiers and large language models — relies heavily on:

1. **Linear Algebra** → representing and transforming data
2. **Calculus** → optimizing models during training
3. **Probability & Statistics** → reasoning under uncertainty

When training a neural network:

* Inputs become vectors and matrices
* Predictions are computed using matrix multiplication
* Errors are minimized using gradients and derivatives
* Outputs are interpreted probabilistically

Without mathematical intuition, ML becomes memorizing APIs.
With mathematical understanding, you can:

* Debug models properly
* Understand why architectures work
* Read research papers
* Optimize systems efficiently
* Build models from scratch

---

# Core Concepts at a Glance

| Pillar         | Key Ideas                                               | Where It Appears in ML                            |
| -------------- | ------------------------------------------------------- | ------------------------------------------------- |
| Linear Algebra | Vectors, matrix multiplication, eigendecomposition, SVD | Neural network layers, embeddings, attention, PCA |
| Calculus       | Derivatives, gradients, Jacobians, chain rule           | Backpropagation, optimization                     |
| Probability    | Distributions, Bayes theorem, entropy, KL divergence    | Loss functions, generative models, uncertainty    |

---

# How to Read Any Formula (Quick Beginner Decoder)

When you see a formula, decode it in this order:

1. **What are the inputs?** (vectors, matrices, probabilities, parameters)
2. **What operation is being done?** (add, multiply, sum, gradient, log)
3. **What does the output represent?** (similarity, error, probability, update)
4. **How is this used in ML?** (forward pass, loss, optimization, uncertainty)

Use this template:

> **Formula** → **What it does** → **Tiny numeric example** → **Where it appears in ML**

---

# Before You Start

You should ideally know:

* Basic algebra
* Python functions and loops
* What a derivative means conceptually
* Difference between scalar, vector, and matrix

If not, do not worry. This module explains concepts from intuition first.

---

# Beginner Foundations — Building Intuition

## 1. Vectors — The Coordinates of ML

Imagine a student profile:

| Feature       | Value |
| ------------- | ----- |
| Math score    | 92    |
| Physics score | 88    |
| Attendance    | 95    |

This can be represented as:

$$
\mathbf{v} = [92, 88, 95]
$$

This list of numbers is called a **vector**.

In ML:

* Images become vectors
* Sentences become vectors
* Audio becomes vectors
* User behavior becomes vectors

A vector is simply a numerical representation of something.

---

## 2. Matrices — Collections of Data

One student:

$$
[92, 88, 95]
$$

Entire classroom:

$$
\begin{bmatrix}
92 & 88 & 95 \
75 & 81 & 90 \
84 & 79 & 91
\end{bmatrix}
$$

This is a **matrix**.

Matrices allow computers to process many inputs simultaneously.

This is why GPUs are excellent for AI:

* GPUs are optimized for massive matrix operations
* Neural networks mostly perform repeated matrix multiplication

---

## 3. Calculus — Learning Through Error Reduction

Suppose a model predicts house prices.

If predictions are wrong, we need to know:

* How wrong?
* In which direction should parameters change?
* By how much?

Derivatives answer these questions.

Think of optimization like walking downhill in fog:

* Gradient = slope direction
* Gradient descent = repeatedly stepping downhill
* Lowest valley = minimum error

---

## 4. Probability — Handling Uncertainty

Real-world data is noisy.

A spam classifier cannot be 100% certain.
Instead it predicts:

$$
P(\text{spam}) = 0.97
$$

Probability allows ML systems to:

* Estimate confidence
* Handle uncertainty
* Learn patterns statistically
* Make robust predictions

---

# 1. Linear Algebra

Linear algebra is the language of machine learning.

Almost every ML operation can be represented using vectors and matrices.

---

# 1.1 Scalars, Vectors, Matrices, Tensors

| Object | Notation                        | Shape            | Example         |
| ------ | ------------------------------- | ---------------- | --------------- |
| Scalar | $x \in \mathbb{R}$              | `()`             | `3.14`          |
| Vector | $\mathbf{v} \in \mathbb{R}^n$   | `(n,)`           | `[1,2,3]`       |
| Matrix | $A \in \mathbb{R}^{m \times n}$ | `(m,n)`          | `[[1,2],[3,4]]` |
| Tensor | $\mathcal{T}$                   | `(d1,d2,...,dk)` | batch of images |

---

## Understanding Tensors Intuitively

* Scalar → single number
* Vector → list of numbers
* Matrix → grid of numbers
* Tensor → higher-dimensional extension

Example:

RGB image:

$$
(Height, Width, Channels)
$$

Batch of images:

$$
(Batch, Height, Width, Channels)
$$

Deep learning frameworks like PyTorch and TensorFlow primarily work with tensors.

---

# 1.2 Vector Operations

## Vector Addition

$$
\mathbf{u} + \mathbf{v} = [u_1 + v_1, u_2 + v_2, ..., u_n + v_n]
$$

Example:

$$
[1,2,3] + [4,5,6] = [5,7,9]
$$

---

## Scalar Multiplication

$$
c\mathbf{v} = [cv_1, cv_2, ..., cv_n]
$$

Example:

$$
2[1,2,3] = [2,4,6]
$$

---

## Dot Product (Inner Product)

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = \mathbf{u}^T\mathbf{v}
$$

Example:

$$
[1,2,3] \cdot [4,5,6]
$$

$$
= 1(4) + 2(5) + 3(6)
$$

$$
= 32
$$

What it does:

* Multiplies matching elements and adds them.
* Returns **one number** measuring alignment/similarity.

ML example:

* In a recommender system, user embedding `u` and movie embedding `m` are dotted.
* Larger dot product → stronger predicted preference.

---

## Geometric Meaning of Dot Product

$$
\mathbf{u} \cdot \mathbf{v} = ||\mathbf{u}|| ||\mathbf{v}|| \cos\theta
$$

Interpretation:

| Angle       | Meaning             |
| ----------- | ------------------- |
| $0^\circ$   | Maximum similarity  |
| $90^\circ$  | Independent         |
| $180^\circ$ | Opposite directions |

---

## Why Dot Products Matter in AI

Dot products are everywhere:

* Attention scores in Transformers
* Similarity search
* Recommendation systems
* Neural network weighted sums
* Embedding retrieval

ChatGPT itself heavily relies on dot products between embeddings.

---

# 1.3 Norms

Norms measure vector magnitude.

## p-Norm

$$
\|\mathbf{v}\|_p = \left(\sum_{i=1}^{n}|v_i|^p\right)^{1/p}
$$

What it does:

* Measures vector size/magnitude in different ways depending on `p`.

---

## Common Norms

| Norm | Formula | What it means | ML use case |
| ---- | ------- | ------------- | ----------- |
| L1   | $\|v\|_1=\sum_i |v_i|$ | Total absolute size | Lasso, sparse features |
| L2   | $\|v\|_2=\sqrt{\sum_i v_i^2}$ | Euclidean length | Ridge, weight decay |
| L$\infty$   | $\|v\|_\infty=\max_i |v_i|$ | Largest component only | Robust constraints |

---

## L1 vs L2 Intuition

### L1 Norm

Encourages many weights to become exactly zero.

Useful in:

* Feature selection
* Sparse models
* Lasso regression

### L2 Norm

Penalizes large weights smoothly.

Useful in:

* Ridge regression
* Preventing overfitting
* Stable optimization

---

## Cosine Similarity

$$
\text{cos_sim}(\mathbf{u},\mathbf{v}) = \frac{\mathbf{u}\cdot\mathbf{v}}{||\mathbf{u}||_2||\mathbf{v}||_2}
$$

Range:

$$
[-1,1]
$$

Used in:

* Semantic search
* Vector databases
* RAG systems
* Embedding similarity

Quick numeric example:

* $u=[1,1],\ v=[2,2]$ → cosine similarity = 1 (same direction)
* $u=[1,0],\ v=[0,1]$ → cosine similarity = 0 (orthogonal)

ML example:

* Semantic search compares query embedding with document embeddings via cosine similarity.
* Highest score → most relevant document.

---

## Vector Projection

$$
\text{proj}_{\mathbf{v}}\mathbf{u} = \frac{\mathbf{u}\cdot\mathbf{v}}{\mathbf{v}\cdot\mathbf{v}}\mathbf{v}
$$

Projection tells how much of one vector lies along another vector.

---

# 1.4 Matrix Operations

## Matrix Multiplication

If:

$$
A \in \mathbb{R}^{m \times k}
$$

and

$$
B \in \mathbb{R}^{k \times n}
$$

then:

$$
C = AB
$$

where:

$$
C_{ij} = \sum_{l=1}^{k}A_{il}B_{lj}
$$

What it does:

* Combines rows of `A` with columns of `B` using dot products.
* This is the core computation in neural network layers.

Tiny example:

$$
\begin{bmatrix}1 & 2\end{bmatrix}
\begin{bmatrix}3\\4\end{bmatrix}=11
$$

ML example:

* Dense layer: `y = XW + b`
* `X` = batch of inputs, `W` = learned weights.

---

## Dimension Rule

$$
(m \times k)(k \times n) \rightarrow (m \times n)
$$

Inner dimensions must match.

---

## Important Properties

### Associative

$$
(AB)C = A(BC)
$$

### Distributive

$$
A(B+C)=AB+AC
$$

### Non-Commutative

$$
AB \neq BA
$$

This is extremely important in ML.
Order matters.

---

## Transpose

$$
(AB)^T = B^TA^T
$$

Transpose swaps rows and columns.

---

## Trace

$$
\text{tr}(A)=\sum_i A_{ii}
$$

Trace is the sum of diagonal elements.

ML example:

* In matrix calculus and covariance-based objectives, trace simplifies sums of quadratic terms.

---

## Determinant

For:

$$
\begin{pmatrix}
a & b \
c & d
\end{pmatrix}
$$

$$
\det(A)=ad-bc
$$

Interpretation:

* Measures volume scaling
* Determines invertibility
* Detects collapse of dimensions

If:

$$
\det(A)=0
$$

then matrix is singular and non-invertible.

ML intuition:

* Determinant near 0 means transformation squashes space; information may be lost.

---

# 1.5 Matrix Inverse

For square matrix:

$$
AA^{-1}=I
$$

where:

$$
I = \text{Identity matrix}
$$

---

## Why Inverses Matter

Suppose:

$$
Ax=b
$$

Then:

$$
x=A^{-1}b
$$

This solves systems of equations.

---

## Moore-Penrose Pseudoinverse

Used when matrix is:

* Non-square
* Singular
* Overdetermined

Formula:

$$
A^+=(A^TA)^{-1}A^T
$$

What it does:

* Gives a best-fit inverse when true inverse does not exist.

ML example:

* Linear regression closed-form solution uses this idea for least squares.

Used heavily in:

* Least squares
* Linear regression
* Numerical optimization

---

# 1.6 Eigenvalues & Eigenvectors

Definition:

$$
A\mathbf{v}=\lambda\mathbf{v}
$$

where:

* $\mathbf{v}$ = eigenvector
* $\lambda$ = eigenvalue

---

## Intuition

Normally matrices rotate and stretch vectors.

Eigenvectors are special vectors whose direction remains unchanged after transformation.
Only scaling changes.

---

## Characteristic Equation

$$
\det(A-\lambda I)=0
$$

Solving gives eigenvalues.

---

## Eigendecomposition

$$
A=V\Lambda V^{-1}
$$

where:

* $V$ = eigenvectors
* $\Lambda$ = diagonal matrix of eigenvalues

ML example:

* PCA finds directions (eigenvectors) with largest variance (largest eigenvalues).

---

## Symmetric Matrices

For:

$$
A=A^T
$$

we get:

$$
A=Q\Lambda Q^T
$$

where:

$$
Q^TQ=I
$$

These matrices are extremely important in ML.

Covariance matrices are symmetric.

---

## ML Applications of Eigenvalues

| Application         | Usage                 |
| ------------------- | --------------------- |
| PCA                 | Principal components  |
| Spectral clustering | Graph structure       |
| Stability analysis  | Optimization behavior |
| Google PageRank     | Dominant eigenvector  |

---

# 1.7 Singular Value Decomposition (SVD)

Any matrix:

$$
A \in \mathbb{R}^{m\times n}
$$

can be decomposed as:

$$
A=U\Sigma V^T
$$

---

## Components of SVD

| Matrix   | Meaning                |
| -------- | ---------------------- |
| $U$      | Left singular vectors  |
| $\Sigma$ | Singular values        |
| $V^T$    | Right singular vectors |

---

## Why SVD is Important

SVD works for ANY matrix.

Unlike eigendecomposition, matrix does not need to be square.

---

## Truncated SVD

$$
A \approx U_k\Sigma_kV_k^T
$$

What it does:

* Keeps only top `k` singular values/vectors.
* Compresses data while preserving most important structure.

ML example:

* In recommendation systems, user-item matrix factorization uses low-rank structure.

This reduces dimensionality while preserving maximum information.

---

## Applications of SVD

* PCA
* Recommendation systems
* Compression
* Latent semantic analysis
* Noise reduction
* Matrix factorization

---

> **Run:** `python src/01-math/matrix_ops.py`

This demonstrates:

* Eigendecomposition
* SVD
* Pseudoinverse

---

# 2. Calculus for ML

Calculus allows models to learn.

Without derivatives, neural networks cannot optimize themselves.

---

# 2.1 Partial Derivatives

For:

$$
f(x,y)=x^2y+3xy^2
$$

Partial derivative with respect to $x$:

$$
\frac{\partial f}{\partial x}=2xy+3y^2
$$

Partial derivative with respect to $y$:

$$
\frac{\partial f}{\partial y}=x^2+6xy
$$

ML example:

* If loss depends on multiple weights, partial derivative tells effect of changing **one weight** while others stay fixed.

---

## Intuition

When taking partial derivative with respect to one variable:

* Treat other variables as constants

---

# 2.2 Gradient

Gradient is vector of all partial derivatives.

$$
\nabla_xf=
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \
\frac{\partial f}{\partial x_2} \
\vdots \
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

---

## Important Property

$$
\nabla f
$$

points toward steepest ascent.

Therefore:

$$
-\nabla f
$$

points toward steepest descent.

This is the basis of optimization.

ML example:

* During training, gradient gives direction to update all model parameters to reduce loss.

---

# 2.3 Chain Rule

For composite function:

$$
f(g(x))
$$

$$
\frac{d}{dx}f(g(x))=f'(g(x))g'(x)
$$

---

## Why Chain Rule Matters

Neural networks are nested functions.

Example:

$$
\text{Input} \rightarrow \text{Layer 1} \rightarrow \text{Layer 2} \rightarrow \text{Loss}
$$

Backpropagation repeatedly applies chain rule through all layers.

Tiny ML example:

* If $L$ depends on prediction $\hat{y}$ and $\hat{y}$ depends on weight $w$,
  $$\frac{dL}{dw}=\frac{dL}{d\hat{y}}\cdot\frac{d\hat{y}}{dw}$$
* This is exactly how backprop computes gradients.

---

# 2.4 Jacobian and Hessian

## Jacobian

For:

$$
\mathbf{f}:\mathbb{R}^n\rightarrow\mathbb{R}^m
$$

$$
J_{ij}=\frac{\partial f_i}{\partial x_j}
$$

Jacobian stores first-order derivatives.

---

## Hessian

$$
H_{ij}=\frac{\partial^2f}{\partial x_i\partial x_j}
$$

Hessian stores second-order derivatives.

---

## Hessian Interpretation

| Eigenvalues | Meaning       |
| ----------- | ------------- |
| Positive    | Local minimum |
| Negative    | Local maximum |
| Mixed       | Saddle point  |

---

# 2.5 Gradient Descent

Goal:

$$
\min \mathcal{L}(w)
$$

Update rule:

$$
w_{t+1}=w_t-\eta\nabla_w\mathcal{L}(w_t)
$$

where:

* $\eta$ = learning rate

What it does:

* Moves parameters a small step opposite gradient to reduce loss.

Tiny numeric example:

* If current weight $w=2$, gradient $=0.5$, learning rate $\eta=0.1$:
	$$w_{new}=2-0.1(0.5)=1.95$$

---

## Gradient Descent Intuition

1. Compute error
2. Compute gradient
3. Move opposite gradient
4. Repeat

Eventually model reaches lower loss.

---

## Types of Gradient Descent

| Type           | Uses           |
| -------------- | -------------- |
| Batch GD       | Entire dataset |
| SGD            | One sample     |
| Mini-batch SGD | Small subsets  |

Mini-batch SGD is most common in deep learning.

---

## Numerical Gradient Checking

$$
\frac{\partial f}{\partial x_i}
\approx
\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}
$$

Used to verify correctness of backpropagation implementations.

---

> **Run:** `python src/01-math/calculus_demo.py`

Demonstrates:

* Numerical gradients
* Analytical gradients
* Gradient descent

---

# 3. Probability & Statistics

Probability is the mathematical framework for uncertainty.

Modern AI systems are probabilistic systems.

---

# 3.1 Foundations

## Sample Space

$$
\Omega
$$

Set of all possible outcomes.

---

## Event

$$
A \subseteq \Omega
$$

Subset of outcomes.

---

## Conditional Probability

$$
P(A|B)=\frac{P(A,B)}{P(B)}
$$

Interpretation:

Probability of A given B already happened.

ML example:

* Spam filtering estimates probability of spam given words in email.

---

## Independence

$$
A \perp B
$$

means:

$$
P(A,B)=P(A)P(B)
$$

---

# 3.2 Bayes' Theorem

$$
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
$$

---

## ML Interpretation

$$
P(\theta|D)=\frac{P(D|\theta)P(\theta)}{P(D)}
$$

| Term       | Meaning                |
| ---------- | ---------------------- |
| Posterior  | Updated belief         |
| Likelihood | Data probability       |
| Prior      | Initial belief         |
| Evidence   | Normalization constant |

Plain meaning:

* **Posterior = Prior updated by observed data**.

---

## Why Bayes Theorem Matters

Used in:

* Bayesian ML
* Naive Bayes classifiers
* Uncertainty estimation
* Probabilistic reasoning

---

# 3.3 Random Variables & Distributions

## Expectation

Discrete:

$$
\mathbb{E}[X]=\sum_xxP(X=x)
$$

Continuous:

$$
\mathbb{E}[X]=\int x p(x)dx
$$

Expectation represents average value.

ML example:

* Expected loss is the average loss over data distribution; training tries to minimize it.

---

## Variance

$$
\text{Var}(X)=\mathbb{E}[X^2]-(\mathbb{E}[X])^2
$$

Variance measures spread.

ML example:

* High variance in model performance can indicate overfitting/instability.

---

## Covariance

$$
\text{Cov}(X,Y)=\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]
$$

Covariance measures relationship between variables.

---

# 3.4 Key Distributions

## Bernoulli Distribution

Binary outcome:

$$
P(X=1)=p
$$

$$
P(X=0)=1-p
$$

Used in:

* Binary classification
* Logistic regression

---

## Gaussian Distribution

$$
p(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

Parameters:

* $\mu$ = mean
* $\sigma^2$ = variance

---

## 68–95–99.7 Rule

For normal distribution:

| Range     | Data Covered |
| --------- | ------------ |
| $1\sigma$ | 68%          |
| $2\sigma$ | 95%          |
| $3\sigma$ | 99.7%        |

---

## Multivariate Gaussian

$$
p(x)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}
$$

Widely used in:

* Gaussian mixture models
* Variational autoencoders
* Kalman filters

---

# 3.5 Maximum Likelihood Estimation (MLE)

Suppose data:

$$
D={x^{(1)},...,x^{(N)}}
$$

Likelihood:

$$
L(\theta)=\prod_{i=1}^{N}p(x^{(i)};\theta)
$$

---

## Log-Likelihood

$$
\ell(\theta)=\sum_{i=1}^{N}\log p(x^{(i)};\theta)
$$

Logarithms make optimization numerically stable.

---

## MLE Objective

$$
\hat{\theta}_{\text{MLE}}=\arg\max_{\theta} \ell(\theta)
$$

---

## Why MLE Matters

Many ML loss functions are derived directly from maximum likelihood.

Examples:

* Cross-entropy
* Logistic regression
* Gaussian regression

---

# 3.6 Cross-Entropy & KL Divergence

## Shannon Entropy

$$
H(P)=-\sum_xP(x)\log P(x)
$$

Measures uncertainty.

Higher entropy:

* More uncertainty
* More randomness

---

## Cross-Entropy

$$
H(P,Q)=-\sum_xP(x)\log Q(x)
$$

Used as classification loss.

---

## Cross-Entropy Loss

$$
\mathcal{L}_{CE}=-\sum_ky_k\log\hat{y}_k
$$

For one-hot labels:

$$
=-\log \hat{y}_{y^*}
$$

Tiny ML example:

* True class probability predicted as $0.9$ → loss $=-\log(0.9)$ (small, good)
* True class probability predicted as $0.1$ → loss $=-\log(0.1)$ (large, bad)

---

## KL Divergence

$$
D_{KL}(P||Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)}
$$

Measures difference between distributions.

---

## Important Properties

$$
D_{KL}(P||Q) \geq 0
$$

and:

$$
D_{KL}(P||Q) \neq D_{KL}(Q||P)
$$

Therefore KL divergence is NOT a true distance metric.

---

## Relationship Between Cross-Entropy and KL

$$
H(P,Q)=H(P)+D_{KL}(P||Q)
$$

When labels are fixed:

Minimizing cross-entropy = minimizing KL divergence.

---

> **Run:** `python src/01-math/probability.py`

Demonstrates:

* Entropy
* KL divergence
* MLE fitting

---

# Summary — Why This Math Powers AI

| ML Concept                  | Mathematical Foundation   |
| --------------------------- | ------------------------- |
| Neural network forward pass | Matrix multiplication     |
| Backpropagation             | Chain rule                |
| Attention mechanism         | Dot product + softmax     |
| PCA                         | Eigendecomposition        |
| Optimization                | Gradient descent          |
| Embeddings                  | Vector spaces             |
| Classification loss         | Cross-entropy             |
| Bayesian learning           | Bayes theorem             |
| Generative models           | Probability distributions |
| Regularization              | L1/L2 norms               |

---

# Common Beginner Mistakes

| Mistake                                                          | Correction                                      |
| ---------------------------------------------------------------- | ----------------------------------------------- |
| Treating vectors as only lists                                   | Vectors represent directions and magnitudes     |
| Confusing matrix multiplication with element-wise multiplication | They are completely different operations        |
| Ignoring dimensions                                              | Shape mismatches are one of the biggest ML bugs |
| Memorizing formulas without intuition                            | Always understand geometric meaning             |
| Thinking probability means certainty                             | Probability models uncertainty                  |

---

# Practical ML Connections

## Transformers

Use:

* Dot products
* Softmax
* Matrix multiplication
* Probability distributions

---

## CNNs

Use:

* Matrix operations
* Gradients
* Convolutions

---

## Recommendation Systems

Use:

* Embeddings
* Cosine similarity
* Matrix factorization

---

## Diffusion Models

Use:

* Gaussian distributions
* KL divergence
* Probability theory

---

# Glossary

| Term          | Meaning                                      |
| ------------- | -------------------------------------------- |
| Scalar        | Single number                                |
| Vector        | Ordered list of numbers                      |
| Matrix        | 2D grid of numbers                           |
| Tensor        | Multi-dimensional array                      |
| Gradient      | Vector of derivatives                        |
| Eigenvector   | Direction preserved after transformation     |
| Eigenvalue    | Scaling factor                               |
| SVD           | Matrix decomposition method                  |
| Entropy       | Measure of uncertainty                       |
| Likelihood    | Probability of data given parameters         |
| Posterior     | Updated probability after observing data     |
| KL Divergence | Difference between probability distributions |

---

# Recommended Resources

## Books

### Mathematics for Machine Learning

Authors:

* Deisenroth
* Faisal
* Ong

Website:

* [https://mml-book.github.io](https://mml-book.github.io)

---

### Introduction to Probability

Authors:

* Blitzstein
* Hwang

Website:

* [https://stat110.net](https://stat110.net)

---

# Video Resources

## 3Blue1Brown — Essence of Linear Algebra

Excellent visual explanation of:

* Vectors
* Matrices
* Eigenvectors
* Transformations

YouTube Playlist:

* [https://www.youtube.com/watch?v=fNk_zzaMoSs](https://www.youtube.com/watch?v=fNk_zzaMoSs)

---

## 3Blue1Brown — Essence of Calculus

Best intuition for:

* Derivatives
* Integrals
* Gradients
* Chain rule

Playlist:

* [https://www.youtube.com/watch?v=WUvTyaaNkzM](https://www.youtube.com/watch?v=WUvTyaaNkzM)

---

## MIT 18.06 — Linear Algebra

Professor:

* Gilbert Strang

MIT OpenCourseWare:

* [https://ocw.mit.edu](https://ocw.mit.edu)

---

# NumPy References

Official documentation:

* [https://numpy.org/doc](https://numpy.org/doc)

Important functions:

| Function          | Purpose             |
| ----------------- | ------------------- |
| `np.dot()`        | Dot product         |
| `np.linalg.inv()` | Matrix inverse      |
| `np.linalg.eig()` | Eigenvalues         |
| `np.linalg.svd()` | SVD                 |
| `np.gradient()`   | Numerical gradients |

---

# Interview Reference — Math for ML

## Why must matrix multiplication dimensions match?

Because multiplication computes dot products between rows and columns.
The lengths must be equal.

---

## Why are eigenvectors important?

They identify important directions in transformed space.
Used in:

* PCA
* Spectral methods
* Stability analysis

---

## Why is SVD powerful?

Because it works for any matrix.
It enables:

* Compression
* Denoising
* Dimensionality reduction
* Recommendation systems

---

## Why is chain rule essential in deep learning?

Backpropagation repeatedly applies chain rule across layers.
Without it, neural networks cannot learn.

---

## Why use cross-entropy for classification?

Because it directly corresponds to maximizing likelihood under probabilistic models.

---

# Cheat Sheet — Math for ML

## Dot Product

$$
a \cdot b = \|a\|\,\|b\|\cos\theta
$$

Meaning:

* Measures similarity and alignment between vectors
* Core operation in neural networks and transformers

---

## Cosine Similarity

$$
	ext{cosine similarity} = \frac{a \cdot b}{\|a\|\,\|b\|}
$$

Meaning:

* Measures directional similarity
* Common in embeddings and semantic search

---

## L2 Norm

$$
||x||_2 = \sqrt{\sum x_i^2}
$$

Meaning:

* Euclidean distance from origin
* Used in regularization and optimization

---

## Eigendecomposition

$$
A = Q\Lambda Q^{-1}
$$

Meaning:

* Decomposes matrix into eigenvectors and eigenvalues
* Used in PCA and spectral analysis

---

## Singular Value Decomposition (SVD)

$$
A = U\Sigma V^T
$$

Meaning:

* Works for any matrix shape
* Used in dimensionality reduction and recommendation systems

---

## Gradient Descent

$$
\theta \leftarrow \theta - \alpha \nabla_\theta L
$$

Meaning:

* Updates parameters to minimize loss
* Backbone of neural network training

---

## Bayes Theorem

$$
P(A \mid B)=\frac{P(B \mid A)P(A)}{P(B)}
$$

Meaning:

* Updates belief after observing evidence
* Foundation of Bayesian learning

---

## Maximum Likelihood Estimation (MLE)

$$
\arg\max_\theta \log P(D \mid \theta)
$$

Meaning:

* Finds parameters most likely to generate observed data

---

## KL Divergence

$$
D_{KL}(P || Q)=\sum P(x)\log\frac{P(x)}{Q(x)}
$$

Meaning:

* Measures difference between probability distributions

---

## Cross-Entropy

$$
H(P,Q)=H(P)+D_{KL}(P || Q)
$$

Meaning:

* Common classification loss function
* Minimizing cross-entropy reduces distribution mismatch

---

# Suggested Learning Path

1. Learn vectors and matrices visually
2. Practice NumPy operations
3. Understand derivatives intuitively
4. Implement gradient descent manually
5. Study probability distributions
6. Learn optimization deeply
7. Move into neural networks

---

# Practice Exercises

## Linear Algebra

1. Compute cosine similarity between two vectors
2. Implement matrix multiplication from scratch
3. Compute eigenvalues of a 2×2 matrix

---

## Calculus

1. Differentiate polynomial functions
2. Implement gradient descent manually
3. Compare numerical vs analytical gradients

---

## Probability

1. Compute entropy of distributions
2. Implement Naive Bayes classifier
3. Calculate KL divergence between two distributions

---

# Final Advice

Do not attempt to memorize all formulas immediately.

Focus on:

* Intuition
* Geometry
* Practical meaning
* ML applications

Mathematics becomes significantly easier once connected to real ML systems.

---

*Next: [Module 02 — ML Basics to Advanced](02-ml-basics.md)*
