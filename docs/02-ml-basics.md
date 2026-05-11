# Module 02 — ML Basics to Advanced

> **Runnable code:** `src/02-ml/`
> ```bash
> cd src/02-ml
> python linear_regression.py
> python logistic_regression.py
> python evaluation.py
> python clustering.py
> python pca.py
> python random_forest.py
> python svm.py
> python gradient_boosting.py
> ```

---

## Prerequisites & Overview

**Prerequisites:** Module 01 (linear algebra + calculus + probability), Python, NumPy. `pip install scikit-learn`.
**Estimated time:** 10–15 hours (8 scripts, each covering a distinct algorithm family)

### Why This Module Matters
Classical ML algorithms dominate production tabular data pipelines. XGBoost wins Kaggle tabular competitions; Random Forests are still default choices for feature-rich datasets. Knowing the math behind each algorithm lets you tune hyperparameters with intent rather than guesswork, diagnose overfitting/underfitting, and answer interview questions about algorithmic trade-offs.

### Learning Path

```
Linear Regression → Logistic Regression   (supervised, parametric)
         ↓
    Evaluation Metrics                     (how to measure everything)
         ↓
  Random Forest → SVM → Gradient Boosting  (non-parametric, ensemble)
         ↓
    PCA → Clustering                        (unsupervised)
```

### Before You Start
- Understand vector dot products and matrix multiplication (Module 01)
- Know what a derivative is and how gradient descent works conceptually (Module 01)
- Know what a probability distribution is and what MLE means (Module 01)
- Have NumPy and scikit-learn installed: `pip install numpy scikit-learn`

### Key Mental Model
Every supervised ML algorithm answers the same question: **find parameters $\theta$ that minimize a loss function $\mathcal{L}(\theta)$ on training data while generalizing to unseen data**. The differences are in: (1) what the loss is, (2) what family of functions $f_\theta$ can represent, (3) how the optimization is solved.

---

## 1. Linear Regression

### 1.1 Problem Setup

Given dataset $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^N$ where $\mathbf{x}^{(i)} \in \mathbb{R}^d$, $y^{(i)} \in \mathbb{R}$:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b = \mathbf{w}^T \tilde{\mathbf{x}}$$

where $\tilde{\mathbf{x}} = [1, x_1, \ldots, x_d]^T$ (bias absorbed into $\mathbf{w}$).

**Loss — Mean Squared Error:**

$$\mathcal{L}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (y^{(i)} - \mathbf{w}^T \mathbf{x}^{(i)})^2 = \frac{1}{N} \|y - X\mathbf{w}\|^2_2$$

### 1.2 Ordinary Least Squares (Closed-Form)

Setting $\nabla_{\mathbf{w}} \mathcal{L} = 0$:

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{2}{N} X^T(X\mathbf{w} - \mathbf{y}) = 0$$

$$\Rightarrow \quad \hat{\mathbf{w}}_{\text{OLS}} = (X^T X)^{-1} X^T \mathbf{y}$$

**Conditions for invertibility:** $X^T X$ is PSD; invertible iff $X$ has full column rank (no multicollinearity, $N \geq d$).

**Complexity:** $O(Nd^2 + d^3)$. For large $d$, use gradient descent instead.

### 1.3 Gradient Descent Solution

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \cdot \frac{2}{N} X^T(X\mathbf{w}_t - \mathbf{y})$$

Mini-batch variant processes $B$ samples per step: $O(Bd)$ per iteration vs $O(Nd^2)$ for OLS.

### 1.4 Regularization

**Ridge (L2)** — adds $\lambda \|\mathbf{w}\|_2^2$:
$$\mathcal{L}_{\text{ridge}} = \|X\mathbf{w} - \mathbf{y}\|^2 + \lambda \|\mathbf{w}\|^2$$
$$\hat{\mathbf{w}}_{\text{ridge}} = (X^TX + \lambda I)^{-1} X^T \mathbf{y}$$

- Always invertible (positive definite) even under multicollinearity.
- Shrinks weights toward zero; does not zero them out.
- Bayesian interpretation: Gaussian prior on $\mathbf{w}$.

**Lasso (L1)** — adds $\lambda \|\mathbf{w}\|_1$:
$$\mathcal{L}_{\text{lasso}} = \|X\mathbf{w} - \mathbf{y}\|^2 + \lambda \|\mathbf{w}\|_1$$

- No closed form; use coordinate descent or subgradient methods.
- Induces sparsity (some $w_i$ become exactly 0) → automatic feature selection.
- Geometric intuition: L1 ball has corners on axes; optimum hits corners.

**Elastic Net** — convex combination: $\lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$

> **Run:** `python src/02-ml/linear_regression.py`

---

## 2. Logistic Regression

### 2.1 Sigmoid & Decision

Binary classification: $y \in \{0, 1\}$. Model the probability:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T\mathbf{x}}}$$

Decision rule: $\hat{y} = \mathbb{1}[\sigma(\mathbf{w}^T\mathbf{x}) \geq 0.5]$ — boundary is $\mathbf{w}^T\mathbf{x} = 0$.

**Sigmoid properties:**
- $\sigma(z) \in (0, 1)$
- $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- $\sigma(-z) = 1 - \sigma(z)$ (antisymmetric)

### 2.2 Binary Cross-Entropy Loss

From MLE: maximizing $\prod_i P(y^{(i)} \mid \mathbf{x}^{(i)})$ under Bernoulli:

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \log \hat{p}^{(i)} + (1-y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$

where $\hat{p}^{(i)} = \sigma(\mathbf{w}^T \mathbf{x}^{(i)})$.

**Gradient** (elegant form after chain rule + sigmoid derivative):

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} X^T (\hat{\mathbf{p}} - \mathbf{y})$$

No closed form (non-linear in $\mathbf{w}$) → use gradient descent. Loss is convex → guaranteed global optimum.

### 2.3 Multiclass — Softmax Regression

$$P(y=k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$$

Loss: categorical cross-entropy $\mathcal{L} = -\frac{1}{N}\sum_i \log P(y^{(i)} \mid \mathbf{x}^{(i)})$

> **Run:** `python src/02-ml/logistic_regression.py`

---

## 3. Model Evaluation

### 3.1 Bias-Variance Decomposition

For any estimator $\hat{f}$, the expected MSE at a point $\mathbf{x}$ decomposes as:

$$\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \underbrace{(\mathbb{E}[\hat{f}] - f^*)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f} - \mathbb{E}[\hat{f}])^2]}_{\text{Variance}} + \underbrace{\sigma^2_\epsilon}_{\text{Irreducible noise}}$$

| Model | Bias | Variance | Fixes |
|-------|------|----------|-------|
| Underfitting (too simple) | High | Low | More features, higher complexity |
| Overfitting (too complex) | Low | High | Regularization, more data, dropout |
| Ideal | Low | Low | Model selection, cross-validation |

### 3.2 K-Fold Cross-Validation

Partition data into $K$ equal folds. For each fold $k$:
1. Train on $K-1$ folds → evaluate on fold $k$
2. Average $K$ validation scores

$$\text{CV score} = \frac{1}{K} \sum_{k=1}^K \text{score}_k, \quad \text{std} = \sqrt{\frac{1}{K}\sum_k (\text{score}_k - \overline{\text{score}})^2}$$

$K=5$ or $K=10$ standard. Stratified K-fold preserves class ratios.

### 3.3 Classification Metrics

**Confusion matrix** (binary):

|  | Predicted + | Predicted - |
|--|------------|------------|
| **Actual +** | TP | FN |
| **Actual -** | FP | TN |

$$\text{Accuracy} = \frac{TP + TN}{N}, \quad \text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}$$

$$F_1 = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}} = \frac{2TP}{2TP + FP + FN}$$

$$F_\beta = \frac{(1+\beta^2) \cdot \text{Prec} \cdot \text{Rec}}{\beta^2 \cdot \text{Prec} + \text{Rec}}$$

$\beta > 1$: recall-weighted (medical screening). $\beta < 1$: precision-weighted (spam filter).

**ROC-AUC:** Plots TPR vs FPR across all thresholds. AUC=1 → perfect; AUC=0.5 → random. Threshold-independent.

**PR-AUC:** Precision-Recall curve area. Better for imbalanced datasets (AUC-ROC inflated by many true negatives).

> **Run:** `python src/02-ml/evaluation.py`

---

## 4. Unsupervised Learning

### 4.1 K-Means Clustering

**Objective:** Minimize within-cluster sum of squares (WCSS):

$$\min_{\{C_k\}, \{\boldsymbol{\mu}_k\}} \sum_{k=1}^K \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$

**Lloyd's Algorithm:**
1. Initialize $K$ centroids $\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K$ (random or K-Means++)
2. **Assignment:** $C_k = \{\mathbf{x} : k = \arg\min_j \|\mathbf{x} - \boldsymbol{\mu}_j\|^2\}$
3. **Update:** $\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{\mathbf{x} \in C_k} \mathbf{x}$
4. Repeat until convergence (assignments stop changing)

**Convergence:** Monotonically decreasing WCSS; guaranteed to converge, but may reach local minimum.

**K-Means++ initialization** — probability of choosing next centroid $\propto d^2(\mathbf{x}, \text{nearest existing centroid})$. Reduces iterations, better optima.

**Elbow method:** Plot WCSS vs $K$; choose $K$ where marginal gain drops ("elbow").

### 4.2 DBSCAN

Density-Based Spatial Clustering. Parameters: $\epsilon$ (radius), $\text{MinPts}$ (min neighbors).

- **Core point:** $|\mathcal{N}_\epsilon(\mathbf{x})| \geq \text{MinPts}$
- **Border point:** in neighborhood of core point, fewer than MinPts neighbors
- **Noise/outlier:** neither core nor border

**Key advantages over K-Means:**
- Arbitrary cluster shapes (not just convex)
- Automatically determines $K$
- Explicitly marks outliers
- Invariant to cluster order

> **Run:** `python src/02-ml/clustering.py`

---

## 5. Principal Component Analysis (PCA)

### 5.1 Derivation

**Goal:** Find directions $\mathbf{u}_1, \ldots, \mathbf{u}_k$ (orthonormal) that maximize variance of projected data.

**PC1** maximizes $\text{Var}(\mathbf{u}^T \mathbf{x})$ subject to $\|\mathbf{u}\| = 1$:

$$\text{Var}(\mathbf{u}^T \mathbf{x}) = \mathbf{u}^T \Sigma \mathbf{u}$$

Via Lagrange multipliers: $\Sigma \mathbf{u} = \lambda \mathbf{u}$ — i.e., $\mathbf{u}$ is an eigenvector of $\Sigma$.

Maximum at $\lambda_1$ (largest eigenvalue), $\mathbf{u}_1$ = corresponding eigenvector.

**Algorithm:**
1. Center: $\tilde{X} = X - \bar{\mathbf{x}}^T$
2. Covariance: $\Sigma = \frac{1}{N-1} \tilde{X}^T \tilde{X}$
3. Eigendecompose: $\Sigma = Q \Lambda Q^T$, sort by $\lambda$ descending
4. Project: $Z = \tilde{X} Q_k$ (keep top $k$ eigenvectors)

**Equivalently via SVD:** $\tilde{X} = U\Sigma V^T$ → principal components = rows of $V^T$.

**Explained variance ratio** for component $j$:
$$\text{EVR}_j = \frac{\lambda_j}{\sum_i \lambda_i}$$

**Choosing $k$:** Select smallest $k$ such that $\sum_{j=1}^k \text{EVR}_j \geq 0.95$.

**Reconstruction:**
$$\tilde{X} \approx Z Q_k^T = \tilde{X} Q_k Q_k^T$$

Reconstruction error = variance in discarded components = $\sum_{j=k+1}^d \lambda_j$.

> **Run:** `python src/02-ml/pca.py`

---

## 6. Decision Trees

### 6.1 Splitting Criteria

A node splits feature $j$ at threshold $t$, creating left ($x_j \leq t$) and right ($x_j > t$) children.

**Gini impurity** (used by CART):
$$G(S) = 1 - \sum_{k=1}^K p_k^2$$

Ranges $[0, 1 - 1/K]$. Zero for pure node.

**Entropy** (used by ID3/C4.5):
$$H(S) = -\sum_{k=1}^K p_k \log_2 p_k$$

**Information Gain:**
$$\text{IG}(S, j, t) = H(S) - \frac{|S_L|}{|S|}H(S_L) - \frac{|S_R|}{|S|}H(S_R)$$

**For regression:** minimize weighted MSE of children.

### 6.2 Overfitting & Pruning

Trees overfit easily (zero training error possible). Controls:
- `max_depth`: limit tree depth
- `min_samples_split` / `min_samples_leaf`: require minimum samples at nodes
- `min_impurity_decrease`: only split if IG exceeds threshold
- **Post-pruning (cost-complexity):** $\mathcal{L}(\alpha, T) = \text{MSE}(T) + \alpha |T|$ where $|T|$ = number of leaves. Cross-validate $\alpha$.

---

## 7. Random Forests

### 7.1 Bagging (Bootstrap Aggregating)

Train $M$ models, each on bootstrap sample (sample $N$ with replacement from $N$ training points):
$$\hat{f}(x) = \frac{1}{M} \sum_{m=1}^M \hat{f}_m(x) \quad \text{(regression)}$$
$$\hat{y} = \text{majority vote} \{f_m(x)\}_{m=1}^M \quad \text{(classification)}$$

**Variance reduction:** For $M$ uncorrelated models each with variance $\sigma^2$:
$$\text{Var}\!\left(\frac{1}{M}\sum_m f_m\right) = \frac{\sigma^2}{M}$$

But trees trained on same data are correlated → Random Forests decorrelate by randomly subsetting features.

### 7.2 Random Forest Algorithm

At each node split, consider only $m \leq d$ randomly selected features:
- Classification: $m = \lfloor\sqrt{d}\rfloor$
- Regression: $m = \lfloor d/3 \rfloor$

This + bagging → low correlation between trees → strong variance reduction.

**Out-of-Bag (OOB) Error:** Each bootstrap sample leaves out ~$1/e \approx 36.8\%$ of data. Use those samples as validation → free cross-validation estimate without separate holdout.

### 7.3 Feature Importance

**Mean Decrease in Impurity (MDI):**
$$\text{FI}_j = \frac{1}{M} \sum_{m=1}^M \sum_{\text{node } t \text{ splits on } j} \frac{N_t}{N} \cdot \Delta\text{impurity}(t)$$

Weighted by fraction of samples reaching node. Can be biased toward high-cardinality features.

**Permutation importance:** Shuffle feature $j$ in validation set, measure accuracy drop. Model-agnostic, unbiased.

> **Run:** `python src/02-ml/random_forest.py`

---

## 8. Support Vector Machines

### 8.1 Hard-Margin SVM (Linearly Separable)

Find hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$ that maximizes the margin $\frac{2}{\|\mathbf{w}\|}$.

**Primal:**
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1 \; \forall i$$

### 8.2 Soft-Margin SVM

Allow misclassification via slack variables $\xi^{(i)} \geq 0$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_i \xi_i \quad \text{s.t.} \quad y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**Hinge loss interpretation:** $\xi_i = \max(0, 1 - y^{(i)} f(\mathbf{x}^{(i)}))$

$$\min_{\mathbf{w}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_i \max(0, 1 - y^{(i)}\mathbf{w}^T\mathbf{x}^{(i)})$$

$C$ controls tradeoff: large $C$ = narrow margin, low training error; small $C$ = wide margin, more misclassifications.

### 8.3 Dual Formulation & Kernel Trick

Lagrangian dual of the primal:

$$\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y^{(i)} y^{(j)} \mathbf{x}^{(i)T}\mathbf{x}^{(j)}$$

$$\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y^{(i)} = 0$$

Decision function: $f(\mathbf{x}) = \text{sign}\!\left(\sum_i \alpha_i y^{(i)} \mathbf{x}^{(i)T}\mathbf{x} + b\right)$

**Kernel trick:** Replace $\mathbf{x}^{(i)T}\mathbf{x}^{(j)}$ with $K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \phi(\mathbf{x}^{(i)})^T\phi(\mathbf{x}^{(j)})$. Compute inner product in high-dim space without explicit $\phi$.

| Kernel | Formula | Use |
|--------|---------|-----|
| Linear | $\mathbf{x}^T\mathbf{x}'$ | Linearly separable, high-dim sparse |
| Polynomial | $(\gamma \mathbf{x}^T\mathbf{x}' + r)^d$ | Polynomial interactions |
| RBF/Gaussian | $\exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$ | General non-linear, most common |
| Sigmoid | $\tanh(\gamma \mathbf{x}^T\mathbf{x}' + r)$ | Neural network-like |

**Mercer's theorem:** $K$ is a valid kernel iff the kernel matrix $K_{ij} = K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ is PSD for all datasets.

> **Run:** `python src/02-ml/svm.py`

---

## 9. Gradient Boosting

### 9.1 Framework

Build additive ensemble sequentially:
$$F_M(\mathbf{x}) = \sum_{m=0}^M \eta \cdot h_m(\mathbf{x})$$

where $h_m$ is a weak learner (typically shallow tree) trained on the negative gradient (pseudo-residuals) of the loss at step $m-1$.

### 9.2 Mathematical Derivation

At step $m$, fit $h_m$ to residuals $r^{(i)}_m = -\frac{\partial \mathcal{L}(y^{(i)}, F_{m-1}(\mathbf{x}^{(i)}))}{\partial F_{m-1}(\mathbf{x}^{(i)})}$

**For MSE loss** ($\mathcal{L} = \frac{1}{2}(y-F)^2$):
$$r^{(i)}_m = y^{(i)} - F_{m-1}(\mathbf{x}^{(i)})$$
Residuals = actual residuals. Each tree fits the remaining unexplained signal.

**For log-loss** (binary classification):
$$r^{(i)}_m = y^{(i)} - \sigma(F_{m-1}(\mathbf{x}^{(i)}))$$

### 9.3 XGBoost Innovations

**Regularized objective:**
$$\mathcal{L}^{(m)} = \sum_i \ell(y^{(i)}, \hat{y}^{(i)}_{m-1} + h_m(\mathbf{x}^{(i)})) + \Omega(h_m)$$

where $\Omega(h) = \gamma T + \frac{1}{2}\lambda \|\mathbf{w}\|^2$ ($T$ = number of leaves, $\mathbf{w}$ = leaf weights).

**2nd-order Taylor expansion:**
$$\mathcal{L}^{(m)} \approx \sum_i [g_i h_m(\mathbf{x}^{(i)}) + \frac{1}{2} h_i^{(i)} h_m^2(\mathbf{x}^{(i)})] + \Omega(h_m)$$

where $g_i = \partial_{\hat{y}} \ell$, $h_i = \partial^2_{\hat{y}} \ell$ (gradient and hessian).

**Optimal leaf weight** for leaf $j$: $w_j^* = -\frac{\sum_{i \in j} g_i}{\sum_{i \in j} h_i + \lambda}$

**Split gain** for splitting leaf $j$ → left $L$, right $R$:
$$\text{Gain} = \frac{1}{2}\left[\frac{(\sum_{i\in L}g_i)^2}{\sum_{i\in L}h_i + \lambda} + \frac{(\sum_{i\in R}g_i)^2}{\sum_{i\in R}h_i + \lambda} - \frac{(\sum_{i\in j}g_i)^2}{\sum_{i\in j}h_i + \lambda}\right] - \gamma$$

Split only if $\text{Gain} > 0$.

**Key XGBoost features:** Column subsampling, row subsampling, approximate split finding, sparsity-aware algorithm, cache-aware block structure.

> **Run:** `python src/02-ml/gradient_boosting.py`

---

## Summary: Algorithm Selection Guide

| Scenario | Algorithm | Why |
|----------|-----------|-----|
| Baseline regression | Linear Regression + Ridge | Fast, interpretable, handles multicollinearity |
| Binary classification baseline | Logistic Regression | Probabilistic, fast, L1 for sparsity |
| Non-linear, tabular data | Random Forest | Robust, low hyperparameter sensitivity |
| Best tabular performance | XGBoost / LightGBM | State of art on structured data |
| High-dim, small dataset | SVM + RBF | Effective in high-dim spaces |
| Imbalanced classes | Any + class_weight / SMOTE | Adjust decision threshold |
| Clustering, unknown K | DBSCAN | Handles arbitrary shapes, noise |
| Dimensionality reduction | PCA | Linear; use UMAP for non-linear |

---

## Resources

### Books
- **The Elements of Statistical Learning** — Hastie, Tibshirani, Friedman. Free PDF at `web.stanford.edu/~hastie/ElemStatLearn/`. The definitive reference for all algorithms in this module.
- **Introduction to Statistical Learning** (ISL) — James et al. More accessible version. Free PDF at `statlearning.com`.
- **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** — Aurélien Géron. Best practical Python companion.

### Video & Courses
- **Andrew Ng — Machine Learning Specialization** (Coursera): The best structured entry point. Covers linear/logistic regression and trees. Free to audit.
- **StatQuest with Josh Starmer** (YouTube): Short, clear videos for every algorithm in this module. Search "StatQuest Random Forest", "StatQuest SVM", etc.
- **fast.ai Practical ML for Coders**: Tabular focus, ensemble methods, feature engineering.

### Papers
- **XGBoost: A Scalable Tree Boosting System** — Chen & Guestrin (2016): `arxiv.org/abs/1603.02754`
- **A Training Algorithm for Optimal Margin Classifiers** — Boser, Guyon, Vapnik (1992): Original SVM paper.
- **Random Forests** — Breiman (2001): Machine Learning journal. Foundational ensemble paper.

### Tooling
- scikit-learn user guide (`scikit-learn.org/stable/user_guide.html`): Algorithm-by-algorithm reference with worked examples.
- XGBoost docs (`xgboost.readthedocs.io`): Production gradient boosting.

---

*Next: [Module 03 — Databases & Vector DBs](03-databases.md)*
