# Module 02 — ML Basics to Advanced

> **Runnable code:** `src/02-ml/`
> ```bash
> python src/02-ml/linear_regression.py
> python src/02-ml/logistic_regression.py
> python src/02-ml/evaluation.py
> python src/02-ml/clustering.py
> python src/02-ml/pca.py
> python src/02-ml/random_forest.py
> python src/02-ml/svm.py
> python src/02-ml/gradient_boosting.py
> python src/02-ml/decision_tree.py
> ```

---

## Prerequisites & Overview

**Prerequisites:** Module 01 (vectors, matrix multiply, gradient descent, probability). `pip install numpy scikit-learn`.
**Estimated time:** 12–18 hours

**Install:**
```bash
pip install numpy scikit-learn
```

### Why This Module Matters

Classical ML algorithms dominate **production tabular data pipelines**. XGBoost wins Kaggle competitions; Random Forests are the default choice for feature-rich datasets. Knowing the math lets you:
- Tune hyperparameters with intent rather than guesswork
- Diagnose overfitting vs underfitting
- Choose the right algorithm for each problem
- Answer every "how does X work" interview question

### Learning Path

```
Linear Regression → Logistic Regression   (supervised, parametric)
         ↓
    Evaluation Metrics                     (how to measure performance)
         ↓
  Decision Trees → Random Forests          (non-parametric, ensemble)
         ↓
    SVM → Gradient Boosting                (margin-based, boosting)
         ↓
    PCA → K-Means                          (unsupervised)
```

### The Universal ML Framework

Every supervised algorithm answers the same question:

> Find parameters $\theta$ that minimize a loss function $\mathcal{L}(\theta)$ on training data while generalizing to unseen data.

Differences between algorithms: (1) what the loss is, (2) what functions $f_\theta$ can represent, (3) how optimization is solved.

---

# 1. Linear Regression

## Intuition

You have house data: square footage, number of rooms, age. You want to **predict price**. Linear regression says: "price is a weighted sum of features."

## 1.1 Model Setup

Given dataset $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^N$ where $\mathbf{x}^{(i)} \in \mathbb{R}^d$, $y^{(i)} \in \mathbb{R}$:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b$$

**Loss — Mean Squared Error:**
$$\mathcal{L}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (y^{(i)} - \hat{y}^{(i)})^2 = \frac{1}{N} \|y - X\mathbf{w}\|^2_2$$

### Code: From Data to Prediction

```python
import numpy as np

rng = np.random.default_rng(42)

# ── Generate synthetic house data ──────────────────────────
n = 150
sqft    = rng.uniform(500, 3000, n)
rooms   = rng.integers(1, 6, n).astype(float)
age     = rng.uniform(0, 50, n)

# True relationship: price = 100*sqft + 20000*rooms - 500*age + 50000 + noise
X = np.column_stack([sqft, rooms, age])
y = 100*sqft + 20000*rooms - 500*age + 50000 + rng.normal(0, 10000, n)

print(f"Dataset: {n} houses, {X.shape[1]} features")
print(f"Price range: ${y.min():,.0f} — ${y.max():,.0f}")
print(f"\nSample row (sqft, rooms, age): {X[0]}")
print(f"Actual price: ${y[0]:,.0f}")
```

## 1.2 Ordinary Least Squares (Closed-Form)

Set gradient to zero and solve analytically:

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{2}{N} X^T(X\mathbf{w} - \mathbf{y}) = 0$$

$$\hat{\mathbf{w}}_{\text{OLS}} = (X^T X)^{-1} X^T \mathbf{y}$$

```python
# Add bias column (column of 1s) to X
X_b = np.column_stack([np.ones(n), X])  # shape (150, 4)

# Closed-form OLS: w = (X^T X)^{-1} X^T y
# np.linalg.lstsq is numerically more stable than explicit inverse
w_ols, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)

print(f"OLS solution (bias, sqft, rooms, age):")
print(f"  bias  : {w_ols[0]:>10.2f}  (true: 50000)")
print(f"  sqft  : {w_ols[1]:>10.2f}  (true: 100)")
print(f"  rooms : {w_ols[2]:>10.2f}  (true: 20000)")
print(f"  age   : {w_ols[3]:>10.2f}  (true: -500)")

# Predictions
y_pred_ols = X_b @ w_ols
mse_ols    = ((y - y_pred_ols)**2).mean()
print(f"\nOLS MSE:  {mse_ols:,.2f}")
print(f"OLS RMSE: ${np.sqrt(mse_ols):,.0f}")
```

**When OLS fails:**
- $N < d$ (more features than samples) → $X^TX$ not invertible
- Multicollinearity (features correlated) → unstable solution
- $N$ very large → $O(Nd^2 + d^3)$ is too slow → use gradient descent

## 1.3 Gradient Descent Solution

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \cdot \frac{2}{N} X^T(X\mathbf{w}_t - \mathbf{y})$$

```python
# Normalize features first (critical for gradient descent convergence)
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_norm = (X - X_mean) / X_std

X_b_norm = np.column_stack([np.ones(n), X_norm])

w = np.zeros(4)
lr = 0.01

for epoch in range(1000):
    y_pred  = X_b_norm @ w
    errors  = y_pred - y
    grad    = (2/n) * (X_b_norm.T @ errors)
    w      -= lr * grad

    if epoch % 200 == 0:
        mse = (errors**2).mean()
        print(f"Epoch {epoch}: MSE={mse:,.0f}")

print(f"\nFinal RMSE: ${np.sqrt((errors**2).mean()):,.0f}")
```

## 1.4 Regularization

**Why it matters:** Without regularization, complex models memorize training noise (overfit).

### Ridge (L2)

$$\mathcal{L}_{\text{ridge}} = \underbrace{\|X\mathbf{w} - \mathbf{y}\|^2}_{\text{fit}} + \underbrace{\lambda \|\mathbf{w}\|^2}_{\text{complexity penalty}}$$

Closed form: $\hat{\mathbf{w}}_{\text{ridge}} = (X^TX + \lambda I)^{-1} X^T \mathbf{y}$ (always invertible!)

### Lasso (L1)

$$\mathcal{L}_{\text{lasso}} = \|X\mathbf{w} - \mathbf{y}\|^2 + \lambda \|\mathbf{w}\|_1$$

No closed form. Forces some weights to exactly zero → automatic feature selection.

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)       # use TRAIN stats on test!

# Compare regularization strengths
print(f"{'Model':<20} {'Train RMSE':>12} {'Test RMSE':>12}")
print("-" * 46)

for alpha in [0.001, 1.0, 100.0, 1000.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_s, y_train)
    train_rmse = np.sqrt(((ridge.predict(X_train_s) - y_train)**2).mean())
    test_rmse  = np.sqrt(((ridge.predict(X_test_s)  - y_test )**2).mean())
    print(f"Ridge α={alpha:<10.3f} {train_rmse:>12,.0f} {test_rmse:>12,.0f}")

print()
for alpha in [1.0, 100.0, 1000.0]:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_s, y_train)
    train_rmse = np.sqrt(((lasso.predict(X_train_s) - y_train)**2).mean())
    test_rmse  = np.sqrt(((lasso.predict(X_test_s)  - y_test )**2).mean())
    nz = (lasso.coef_ != 0).sum()
    print(f"Lasso α={alpha:<10.3f} {train_rmse:>12,.0f} {test_rmse:>12,.0f}  (non-zero: {nz}/3)")
```

---

# 2. Logistic Regression

## Intuition

You want to classify emails as spam or not-spam. The output should be a **probability** (0 to 1), not a raw score. Logistic regression takes the linear score $\mathbf{w}^T\mathbf{x}$ and squashes it into $[0, 1]$ using the sigmoid.

## 2.1 Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b)$$

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Observe the squashing behavior
z_vals = np.array([-5, -2, -1, 0, 1, 2, 5])

print(f"{'z':>5} {'σ(z)':>8} {'Interpretation'}")
print("-" * 40)
for z in z_vals:
    s = sigmoid(z)
    interp = "very likely 1" if s > 0.9 else "likely 1" if s > 0.7 else \
             "uncertain" if 0.3 < s < 0.7 else "likely 0" if s > 0.1 else "very likely 0"
    print(f"{z:>5.0f} {s:>8.4f}  {interp}")

# Sigmoid derivative: σ(z)(1-σ(z))
# Maximum derivative at z=0: σ(0)*(1-σ(0)) = 0.5*0.5 = 0.25
print(f"\nSigmoid at 0: {sigmoid(0):.4f}")
print(f"Max gradient (at z=0): {sigmoid(0)*(1-sigmoid(0)):.4f}")
```

**Decision boundary:** Where $\sigma(\mathbf{w}^T\mathbf{x}) = 0.5$, i.e., $\mathbf{w}^T\mathbf{x} = 0$. This is a hyperplane in feature space.

## 2.2 Binary Cross-Entropy Loss

From MLE under Bernoulli distribution:

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \log \hat{p}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$

**Gradient** (elegant result after chain rule through sigmoid):
$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} X^T (\hat{\mathbf{p}} - \mathbf{y})$$

```python
import numpy as np

rng = np.random.default_rng(42)

# Generate binary classification data: pass/fail based on study+sleep
n = 300
study_h  = rng.uniform(1, 10, n)
sleep_h  = rng.uniform(4,  9, n)

# True boundary: 3*study + 2*sleep > 25 → pass
score = 3 * study_h + 2 * sleep_h + rng.normal(0, 2, n)
y     = (score > 25).astype(float)

print(f"Pass rate: {y.mean():.1%}")

X = np.column_stack([study_h, sleep_h])
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_norm = (X - X_mean) / X_std
X_b    = np.column_stack([np.ones(n), X_norm])

# ── Logistic Regression from scratch ──────────────────────────
w = np.zeros(3)   # [bias, w_study, w_sleep]
lr = 0.1

for epoch in range(500):
    # Forward: compute probabilities
    logits   = X_b @ w          # shape (n,)
    p_hat    = sigmoid(logits)  # shape (n,)

    # Loss: binary cross-entropy
    eps  = 1e-9
    loss = -(y * np.log(p_hat + eps) + (1-y) * np.log(1 - p_hat + eps)).mean()

    # Gradient
    grad = (1/n) * (X_b.T @ (p_hat - y))

    # Update
    w -= lr * grad

    if epoch % 100 == 0:
        # Accuracy
        predictions = (p_hat >= 0.5).astype(float)
        acc = (predictions == y).mean()
        print(f"Epoch {epoch:3d}: loss={loss:.4f}, accuracy={acc:.1%}")

print(f"\nLearned weights: bias={w[0]:.3f}, study={w[1]:.3f}, sleep={w[2]:.3f}")

# Predict new student: 7 hours study, 8 hours sleep
new = np.array([[1.0, (7 - X_mean[0])/X_std[0], (8 - X_mean[1])/X_std[1]]])
prob = sigmoid(new @ w)[0]
print(f"\nNew student (study=7h, sleep=8h): P(pass)={prob:.3f} → {'PASS' if prob>0.5 else 'FAIL'}")
```

## 2.3 Multiclass — Softmax Regression

For $K$ classes, assign a weight vector per class:

$$P(y=k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$$

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris     = load_iris()
X_iris   = iris.data
y_iris   = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# multi_class='multinomial' uses softmax
clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
clf.fit(X_train_s, y_train)

y_pred   = clf.predict(X_test_s)
accuracy = (y_pred == y_test).mean()
print(f"Iris multiclass accuracy: {accuracy:.1%}")

# Probability outputs
probs = clf.predict_proba(X_test_s)[:3]
print(f"\nSample probabilities (3 samples):")
for i, row in enumerate(probs):
    print(f"  Sample {i}: " + ", ".join(f"class{k}={p:.3f}" for k, p in enumerate(row)))
```

---

# 3. Model Evaluation

## Intuition

A model that memorizes training data is useless. Evaluation measures **generalization** — how well the model works on new, unseen data.

## 3.1 Bias-Variance Decomposition

For any estimator $\hat{f}$:

$$\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \underbrace{(\mathbb{E}[\hat{f}] - f^*)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f} - \mathbb{E}[\hat{f}])^2]}_{\text{Variance}} + \underbrace{\sigma^2_\epsilon}_{\text{Noise}}$$

| State | Bias | Variance | Symptom | Fix |
|-------|------|----------|---------|-----|
| Underfitting | High | Low | Train error also high | More features, higher model complexity |
| Overfitting | Low | High | Train ≪ Val error | Regularization, more data, early stopping |
| Ideal | Low | Low | Train ≈ Val, both low | Model selection |

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)

# True function: y = sin(x) + noise
x = np.linspace(0, 3*np.pi, 100)
y_true = np.sin(x)
y_noisy = y_true + rng.normal(0, 0.3, len(x))

x_reshaped = x.reshape(-1, 1)
x_train, x_val, y_train, y_val = train_test_split(
    x_reshaped, y_noisy, test_size=0.3, random_state=42
)

print(f"{'Degree':<8} {'Train RMSE':>12} {'Val RMSE':>12} {'State'}")
print("-" * 50)

for deg in [1, 2, 3, 5, 10, 15]:
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(x_train, y_train)

    train_rmse = np.sqrt(((model.predict(x_train) - y_train)**2).mean())
    val_rmse   = np.sqrt(((model.predict(x_val)   - y_val  )**2).mean())

    if train_rmse > 0.4:
        state = "Underfitting"
    elif val_rmse > train_rmse * 1.5:
        state = "Overfitting"
    else:
        state = "Good"

    print(f"{deg:<8} {train_rmse:>12.4f} {val_rmse:>12.4f} {state}")
```

## 3.2 K-Fold Cross-Validation

Never use the test set for hyperparameter tuning — it leaks information. Use **K-fold** on the training set.

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge

rng = np.random.default_rng(42)
X = rng.randn(200, 5)
y = X @ np.array([1, -2, 0.5, 3, -1]) + rng.normal(0, 1, 200)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("5-Fold Cross-Validation for Ridge (various α):")
print(f"{'Alpha':<10} {'Mean RMSE':>12} {'Std RMSE':>12}")
print("-" * 36)

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    model  = Ridge(alpha=alpha)
    scores = cross_val_score(model, X, y, cv=kf,
                             scoring='neg_mean_squared_error')
    rmses  = np.sqrt(-scores)
    print(f"{alpha:<10.2f} {rmses.mean():>12.4f} {rmses.std():>12.4f}")
```

## 3.3 Classification Metrics

**Confusion matrix:**

|  | Predicted Positive | Predicted Negative |
|--|---|---|
| **Actual Positive** | TP (True Pos) | FN (False Neg) |
| **Actual Negative** | FP (False Pos) | TN (True Neg) |

$$\text{Accuracy} = \frac{TP+TN}{N} \qquad \text{Precision} = \frac{TP}{TP+FP} \qquad \text{Recall} = \frac{TP}{TP+FN}$$

$$F_1 = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}} = \frac{2TP}{2TP+FP+FN}$$

**When accuracy fails:** 99% "not fraud" predictions on a 1%-fraud dataset is 99% accurate but useless. Use F1 or PR-AUC for imbalanced data.

```python
import numpy as np

# Manual confusion matrix and metrics
y_true = np.array([1,1,1,1,1,0,0,0,0,0])
y_pred = np.array([1,1,1,0,0,1,0,0,0,0])

TP = ((y_pred == 1) & (y_true == 1)).sum()
TN = ((y_pred == 0) & (y_true == 0)).sum()
FP = ((y_pred == 1) & (y_true == 0)).sum()
FN = ((y_pred == 0) & (y_true == 1)).sum()

accuracy  = (TP + TN) / len(y_true)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Confusion Matrix:")
print(f"  TP={TP}, TN={TN}, FP={FP}, FN={FN}")
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}  (of all predicted +, how many are really +?)")
print(f"Recall:    {recall:.4f}  (of all actual +, how many did we catch?)")
print(f"F1 score:  {f1:.4f}  (harmonic mean of precision & recall)")
```

### ROC-AUC

```python
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_tr, y_tr)

probs  = model.predict_proba(X_te)[:, 1]  # probability of class 1
auc    = roc_auc_score(y_te, probs)

fpr, tpr, thresholds = roc_curve(y_te, probs)

print(f"AUC-ROC: {auc:.4f}")
print(f"\nROC curve points (sample):")
print(f"  {'Threshold':>10} {'FPR':>8} {'TPR':>8}")
for i in range(0, len(thresholds), len(thresholds)//5):
    print(f"  {thresholds[i]:>10.3f} {fpr[i]:>8.3f} {tpr[i]:>8.3f}")
```

---

# 4. Decision Trees

## Intuition

Think of a decision tree as 20 Questions. You ask yes/no questions to narrow down the answer. At each step, you choose the question that splits the data most cleanly.

## 4.1 Splitting Criteria

**Gini impurity** (CART default):
$$G(S) = 1 - \sum_{k=1}^K p_k^2 \qquad \in [0,\ 1 - 1/K]$$

Zero for pure node (all one class). Maximum when classes are equally distributed.

**Information Gain:**
$$\text{IG}(S, j, t) = H(S) - \frac{|S_L|}{|S|}H(S_L) - \frac{|S_R|}{|S|}H(S_R)$$

```python
import numpy as np

def gini(labels):
    if len(labels) == 0:
        return 0
    classes, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return 1 - (probs**2).sum()

def entropy(labels):
    if len(labels) == 0:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    probs = probs[probs > 0]
    return -(probs * np.log2(probs)).sum()

def information_gain(parent, left, right):
    n = len(parent)
    h_parent = entropy(parent)
    h_left   = entropy(left)
    h_right  = entropy(right)
    return h_parent - (len(left)/n * h_left) - (len(right)/n * h_right)

# Example: splitting students by study hours
labels  = np.array([0, 0, 0, 1, 1, 1, 1, 1])   # 0=fail, 1=pass
left    = labels[:3]   # study < 4 hours → [0, 0, 0]
right   = labels[3:]   # study ≥ 4 hours → [1, 1, 1, 1, 1]

print(f"Parent Gini:        {gini(labels):.4f}")
print(f"Parent Entropy:     {entropy(labels):.4f}")
print(f"Left child Gini:    {gini(left):.4f}   (pure → 0!)")
print(f"Right child Gini:   {gini(right):.4f}  (pure → 0!)")
print(f"Information Gain:   {information_gain(labels, left, right):.4f}")
```

## 4.2 Decision Tree from Scratch (Minimal)

```python
class DecisionNode:
    def __init__(self):
        self.feature    = None   # which feature to split on
        self.threshold  = None   # split value
        self.left       = None   # left child
        self.right      = None   # right child
        self.value      = None   # leaf prediction (if leaf)

class DecisionTreeScratch:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)

    def _build(self, X, y, depth):
        node = DecisionNode()

        # Stop conditions: max depth, pure node, or too few samples
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 2:
            # Leaf: predict most common class
            values, counts = np.unique(y, return_counts=True)
            node.value = values[np.argmax(counts)]
            return node

        # Find best split
        best_gain = -1
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask  = X[:, feat] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gain = information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain       = gain
                    node.feature    = feat
                    node.threshold  = thresh

        if best_gain == 0:
            values, counts = np.unique(y, return_counts=True)
            node.value = values[np.argmax(counts)]
            return node

        left_mask  = X[:, node.feature] <= node.threshold
        node.left  = self._build(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

# Test it
from sklearn.datasets import load_iris
iris = load_iris()
X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target,
                                            test_size=0.2, random_state=42)

tree = DecisionTreeScratch(max_depth=4)
tree.fit(X_tr, y_tr)
acc  = (tree.predict(X_te) == y_te).mean()
print(f"Custom Decision Tree accuracy: {acc:.1%}")
```

## 4.3 Sklearn Decision Tree with Pruning

```python
from sklearn.tree import DecisionTreeClassifier

results = []
for max_depth in [1, 2, 3, 5, 10, None]:
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_tr, y_tr)
    train_acc = dt.score(X_tr, y_tr)
    test_acc  = dt.score(X_te, y_te)
    results.append((max_depth, train_acc, test_acc))
    label = str(max_depth) if max_depth else "None"
    print(f"Depth {label:>5}: train={train_acc:.3f}, test={test_acc:.3f}")
```

---

# 5. Random Forests

## Intuition

One doctor can be wrong. A panel of 100 doctors voting is much more reliable. Random Forests are ensembles of decision trees, each trained on a random subset of data and features.

## 5.1 Bagging (Bootstrap Aggregating)

Each tree trains on a **bootstrap sample** (sample $N$ examples with replacement from $N$ training points). Different trees see different views of the data.

**Why does averaging help?**
- Each tree has variance $\sigma^2$
- $M$ uncorrelated trees averaged: variance = $\sigma^2 / M$
- Trees trained on same data are correlated, so variance reduction is partial
- Random feature subsetting decorrelates trees → stronger variance reduction

```python
import numpy as np

rng = np.random.default_rng(42)

# Simulate bagging: average of M noisy models
true_value = 42.0

def noisy_model():
    return true_value + rng.normal(0, 10)

# Single model estimate
single_errors = [abs(noisy_model() - true_value) for _ in range(1000)]

# Ensemble of 50 models (bagging)
def ensemble(M=50):
    return np.mean([noisy_model() for _ in range(M)])

ensemble_errors = [abs(ensemble(50) - true_value) for _ in range(1000)]

print(f"Single model mean error:   {np.mean(single_errors):.2f}")
print(f"50-model ensemble error:   {np.mean(ensemble_errors):.2f}")
print(f"Improvement factor:         {np.mean(single_errors)/np.mean(ensemble_errors):.1f}x")
```

## 5.2 Random Forest Implementation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20,
                            n_informative=10, n_redundant=5, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=100,    # number of trees
    max_features='sqrt', # features per split: sqrt(n_features) for classification
    max_depth=None,      # grow fully
    min_samples_leaf=1,
    oob_score=True,      # use out-of-bag samples for free validation
    random_state=42
)
rf.fit(X_tr, y_tr)

print(f"Test accuracy:    {rf.score(X_te, y_te):.4f}")
print(f"OOB accuracy:     {rf.oob_score_:.4f}")  # FREE validation estimate!

# Feature importance
importances = rf.feature_importances_
ranked      = np.argsort(importances)[::-1]

print(f"\nTop 5 features:")
for i in ranked[:5]:
    print(f"  Feature {i:2d}: {importances[i]:.4f}")
```

## 5.3 Comparing Single Tree vs Random Forest

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

for name, model in [("Decision Tree", dt), ("Random Forest", rf)]:
    model.fit(X_tr, y_tr)
    train_acc = model.score(X_tr, y_tr)
    test_acc  = model.score(X_te, y_te)
    print(f"{name:<20}: train={train_acc:.4f}, test={test_acc:.4f}, "
          f"overfit gap={train_acc - test_acc:.4f}")
```

---

# 6. Support Vector Machines

## Intuition

Draw a line between two classes. Now push that line as far from both classes as possible. The gap between the line and the nearest points is the **margin**. Maximizing margin leads to better generalization.

## 6.1 Hard-Margin SVM

**Find the widest lane between classes:**

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1 \; \forall i$$

The margin width is $\frac{2}{\|\mathbf{w}\|}$. Minimizing $\|\mathbf{w}\|^2$ is equivalent to maximizing the margin.

## 6.2 Soft-Margin SVM

Real data is noisy — allow some misclassifications with penalty $C$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_i \xi_i \quad \text{s.t.} \quad y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1 - \xi_i$$

- **Large $C$**: narrow margin, low training error (risk overfitting)
- **Small $C$**: wide margin, more misclassifications (better generalization)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

# Non-linearly separable data
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

print("SVM with RBF kernel — effect of C:")
print(f"{'C':>8} {'Train Acc':>12} {'Test Acc':>12}")
print("-" * 34)

for C in [0.01, 0.1, 1, 10, 100]:
    svm = SVC(kernel='rbf', C=C, gamma='scale')
    svm.fit(X_tr_s, y_tr)
    train_acc = svm.score(X_tr_s, y_tr)
    test_acc  = svm.score(X_te_s, y_te)
    print(f"{C:>8.2f} {train_acc:>12.4f} {test_acc:>12.4f}")
```

## 6.3 Kernel Trick

Replace dot product $\mathbf{x}^T\mathbf{x}'$ with $K(\mathbf{x}, \mathbf{x}')$ to implicitly map to higher dimensions:

| Kernel | Formula | Use case |
|--------|---------|---------|
| Linear | $\mathbf{x}^T\mathbf{x}'$ | High-dim sparse data (text) |
| RBF/Gaussian | $\exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$ | General non-linear (default) |
| Polynomial | $(\gamma \mathbf{x}^T\mathbf{x}' + r)^d$ | Polynomial interactions |

```python
kernels  = ['linear', 'rbf', 'poly']
print("\nKernel comparison on moons dataset:")
print(f"{'Kernel':<12} {'Test Accuracy':>14}")
print("-" * 28)
for kernel in kernels:
    svm = SVC(kernel=kernel, C=1.0, gamma='scale')
    svm.fit(X_tr_s, y_tr)
    print(f"{kernel:<12} {svm.score(X_te_s, y_te):>14.4f}")
```

---

# 7. Gradient Boosting & XGBoost

## Intuition

Instead of training many independent trees and averaging (Random Forest), Gradient Boosting trains trees **sequentially**. Each new tree fixes the mistakes of all previous trees.

**Analogy:** A student takes a test. The teacher marks wrong answers. The next study session focuses on correcting exactly those mistakes.

## 7.1 Algorithm

At step $m$, compute pseudo-residuals (negative gradient of loss):
$$r^{(i)}_m = -\frac{\partial \mathcal{L}(y^{(i)}, F_{m-1}(\mathbf{x}^{(i)}))}{\partial F_{m-1}(\mathbf{x}^{(i)})}$$

For MSE loss: $r^{(i)}_m = y^{(i)} - F_{m-1}(\mathbf{x}^{(i)})$ — actual residuals.

Train tree $h_m$ on these residuals. Update: $F_m = F_{m-1} + \eta \cdot h_m$

```python
import numpy as np

rng = np.random.default_rng(42)

# Simple gradient boosting from scratch (MSE, stumps)
n = 200
X_gb = rng.uniform(0, 10, (n, 1))
y_gb = np.sin(X_gb.ravel()) + rng.normal(0, 0.3, n)

# Stumps (depth-1 trees) as weak learners
from sklearn.tree import DecisionTreeRegressor

# Initialize: predict mean
F = np.full(n, y_gb.mean())
learning_rate = 0.1
n_estimators  = 50

trees = []
for m in range(n_estimators):
    # Compute residuals (negative gradient of MSE)
    residuals = y_gb - F

    # Fit a stump to residuals
    stump = DecisionTreeRegressor(max_depth=1)
    stump.fit(X_gb, residuals)
    trees.append(stump)

    # Update predictions
    F += learning_rate * stump.predict(X_gb)

final_mse = ((y_gb - F)**2).mean()
print(f"Gradient boosting MSE after {n_estimators} trees: {final_mse:.6f}")
print(f"Baseline (predict mean) MSE: {((y_gb - y_gb.mean())**2).mean():.6f}")
```

## 7.2 XGBoost Key Innovations

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression

X_r, y_r = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

gbm = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,     # shrinkage: smaller = less overfit, needs more trees
    max_depth=3,           # weak learners (shallow trees)
    subsample=0.8,         # stochastic boosting (row subsampling)
    min_samples_leaf=5,
    random_state=42
)
gbm.fit(X_tr, y_tr)

train_rmse = np.sqrt(((gbm.predict(X_tr) - y_tr)**2).mean())
test_rmse  = np.sqrt(((gbm.predict(X_te) - y_te)**2).mean())
print(f"Gradient Boosting — Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")

# XGBoost style: try different hyperparameters
configs = [
    {'n_estimators': 50,  'learning_rate': 0.5, 'max_depth': 5},
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
    {'n_estimators': 500, 'learning_rate': 0.01,'max_depth': 3},
]

print(f"\n{'Trees':<8} {'LR':<8} {'Depth':<7} {'Test RMSE':>10}")
for cfg in configs:
    m = GradientBoostingRegressor(**cfg, random_state=42)
    m.fit(X_tr, y_tr)
    rmse = np.sqrt(((m.predict(X_te) - y_te)**2).mean())
    print(f"{cfg['n_estimators']:<8} {cfg['learning_rate']:<8} {cfg['max_depth']:<7} {rmse:>10.2f}")
```

---

# 8. Unsupervised Learning

## 8.1 K-Means Clustering

### Intuition

Group data points so that points within a group are similar, and points between groups are different. K-Means finds $K$ centroids and assigns each point to its nearest centroid.

**Lloyd's Algorithm:**
1. Initialize $K$ centroids randomly
2. **Assign** each point to nearest centroid
3. **Update** each centroid to the mean of its assigned points
4. Repeat until assignments don't change

```python
import numpy as np

rng = np.random.default_rng(42)

# K-Means from scratch
def kmeans(X, K, max_iter=100):
    n, d = X.shape

    # Step 1: Initialize centroids randomly (K-Means++ would be better)
    idx      = rng.choice(n, K, replace=False)
    centroids = X[idx].copy()

    for iteration in range(max_iter):
        # Step 2: Assign each point to nearest centroid
        # distances: shape (n, K) — distance from each point to each centroid
        diffs     = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, K, d)
        distances = (diffs**2).sum(axis=2)                              # (n, K)
        labels    = distances.argmin(axis=1)                            # (n,)

        # Step 3: Update centroids
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if (labels == k).sum() > 0 else centroids[k]
            for k in range(K)
        ])

        # Check convergence
        if np.allclose(centroids, new_centroids):
            print(f"Converged at iteration {iteration}")
            break
        centroids = new_centroids

    # WCSS (inertia)
    wcss = sum(((X[labels == k] - centroids[k])**2).sum() for k in range(K))
    return labels, centroids, wcss

# Generate clustered data
blobs = []
true_centers = [(0, 0), (5, 5), (-3, 6)]
for cx, cy in true_centers:
    blobs.append(rng.normal([cx, cy], 1.0, (50, 2)))
X_cl = np.vstack(blobs)

labels, centroids, wcss = kmeans(X_cl, K=3)

print(f"\nCluster sizes: {np.bincount(labels)}")
print(f"WCSS: {wcss:.2f}")
print(f"True centers: {true_centers}")
print(f"Found centers:\n{centroids.round(2)}")
```

### Elbow Method for Choosing K

```python
from sklearn.cluster import KMeans

wcss_values = []
K_range = range(1, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cl)
    wcss_values.append(km.inertia_)

print("\nElbow method:")
print(f"{'K':>4} {'WCSS':>12} {'Drop':>12}")
for i, (k, wcss) in enumerate(zip(K_range, wcss_values)):
    drop = wcss_values[i-1] - wcss if i > 0 else 0
    marker = " ← elbow" if k == 3 else ""
    print(f"{k:>4} {wcss:>12.1f} {drop:>12.1f}{marker}")
```

## 8.2 PCA — Principal Component Analysis

```python
import numpy as np

rng = np.random.default_rng(42)

# PCA from scratch
def pca_from_scratch(X, n_components):
    # Step 1: Center the data
    X_centered = X - X.mean(axis=0)

    # Step 2: Compute covariance matrix
    cov = X_centered.T @ X_centered / (len(X) - 1)

    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # eigh for symmetric matrices

    # Step 4: Sort descending (largest eigenvalue first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Project data
    components = eigenvectors[:, :n_components]
    X_reduced  = X_centered @ components

    # Explained variance
    evr = eigenvalues / eigenvalues.sum()

    return X_reduced, components, evr

# High-dimensional data
n, d = 300, 10
X_hd = rng.randn(n, d)
# Make features correlated: only 3 real signals
true_signal = rng.randn(n, 3)
X_hd = true_signal @ rng.randn(3, d) + rng.randn(n, d) * 0.2

X_pca, components, evr = pca_from_scratch(X_hd, n_components=5)

print("PCA explained variance:")
cumulative = 0
for i, (var, cum) in enumerate(zip(evr[:5], np.cumsum(evr[:5]))):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.1f}%)  cumulative: {cum*100:.1f}%")

print(f"\nOriginal shape: {X_hd.shape}")
print(f"Reduced shape:  {X_pca.shape}")

# Reconstruction
X_approx = X_pca @ components.T + X_hd.mean(axis=0)
recon_error = np.linalg.norm(X_hd - X_approx, 'fro')
print(f"Reconstruction error (k=5): {recon_error:.4f}")
```

---

# 9. Interview Q&A

## Q1: Why is logistic regression a classification algorithm if it outputs a continuous value?

The output $\sigma(\mathbf{w}^T\mathbf{x}) \in (0,1)$ is interpreted as $P(y=1|\mathbf{x})$. Classification is the **decision** — pick class 1 if probability exceeds threshold (usually 0.5). The underlying model is still a linear model in log-odds space: $\log\frac{p}{1-p} = \mathbf{w}^T\mathbf{x}$.

## Q2: Why can't you use MSE loss for logistic regression?

MSE + sigmoid is **non-convex** — multiple local minima, gradient descent may not converge to global optimum. Binary cross-entropy is convex (log of sigmoid), guaranteeing convergence to the global minimum.

## Q3: How does a Random Forest reduce variance without increasing bias?

Each tree has high variance (overfits). Averaging $M$ independent trees reduces variance by $1/M$. Trees aren't fully independent (trained on same distribution), but random feature subsampling decorrelates them. The bias of each tree is preserved in the ensemble because averaging doesn't change the expected prediction.

## Q4: SVM vs Logistic Regression — when to use each?

| Criterion | SVM | Logistic Regression |
|-----------|-----|-------------------|
| Output | Decision only | Calibrated probabilities |
| Kernel | Easily kernelized | Linear only (without tricks) |
| Small data | Strong (max-margin) | May underfit |
| Large data | Slow ($O(n^2)$–$O(n^3)$) | Fast, scalable |
| High-dimensional | Excellent (linear SVM) | Also good |
| Interpretability | Support vectors | Weights directly |

## Q5: Why does boosting reduce bias while bagging reduces variance?

**Bagging:** Averages many high-variance, low-bias models (deep trees). The average has same bias but lower variance.

**Boosting:** Each weak learner (shallow tree) has high bias. Sequential training with residuals corrects systematic errors, reducing bias. Risk: if base learners start overfitting residuals, variance increases → use shrinkage and subsampling.

## Q6: What's the difference between Gini and Entropy for tree splitting?

Computationally, Gini avoids log computation. Both produce similar trees in practice. Gini tends to isolate the most frequent class; entropy is more symmetric. scikit-learn defaults to Gini.

## Q7: Why normalize features before SVM but not necessarily before decision trees?

SVMs optimize a geometric margin defined by Euclidean distances — features on different scales distort the geometry. Decision trees split on one feature at a time using threshold comparisons, which are scale-invariant.

---

# 10. Cheat Sheet

| Algorithm | Loss Function | Optimization | Key Hyperparameters |
|-----------|-------------|-------------|-------------------|
| Linear Regression | MSE $\|Xw-y\|^2$ | OLS or GD | $\lambda$ (regularization) |
| Ridge | MSE + $\lambda\|w\|^2$ | Closed form | $\lambda$ |
| Lasso | MSE + $\lambda\|w\|_1$ | Coordinate descent | $\lambda$ |
| Logistic Reg | BCE | GD | $C$ (inverse $\lambda$) |
| Decision Tree | Gini / Entropy | Greedy splits | `max_depth`, `min_samples_leaf` |
| Random Forest | Tree loss | Bagging + greedy | `n_estimators`, `max_features` |
| SVM | Hinge + $\|w\|^2$ | Quadratic programming | $C$, kernel, $\gamma$ |
| Gradient Boosting | Any differentiable | Sequential residuals | `n_estimators`, `learning_rate`, `max_depth` |
| K-Means | WCSS | Lloyd's (EM) | $K$, init method |
| PCA | Reconstruction error | Eigendecomposition | `n_components` |

---

# MINI-PROJECT — Titanic Survival Predictor

**What you will build:** A complete ML pipeline that predicts Titanic passenger survival. You will implement feature engineering, cross-validation, and compare five algorithms.

**Learning goals:** Every algorithm from this module appears. You will see exactly why some algorithms outperform others on this dataset.

---

## Step 1 — Create the Dataset

```python
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

rng = np.random.default_rng(42)

# Synthetic Titanic-like data
n = 800

# Features
pclass = rng.choice([1, 2, 3], p=[0.2, 0.3, 0.5], size=n)   # ticket class
age    = np.clip(rng.normal(30, 15, n), 1, 80)
sex    = rng.choice([0, 1], p=[0.5, 0.5], size=n)             # 0=male, 1=female
sibsp  = rng.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.1, 0.1], size=n)
fare   = np.where(pclass == 1, rng.uniform(50, 300, n),
         np.where(pclass == 2, rng.uniform(10, 50, n),
                               rng.uniform(5, 20, n)))

# Survival: women, first class, children → higher survival
logit = (-1.0
         + 1.5 * sex              # women survived more
         - 0.5 * (pclass - 1)    # 1st class advantage
         - 0.01 * (age - 30)     # children slightly favored
         - 0.2 * sibsp           # too many siblings = chaos
         + rng.normal(0, 0.5, n))

survival = (logit > 0).astype(int)

X_raw = np.column_stack([pclass, age, sex, sibsp, fare])
y     = survival

print(f"Dataset: {n} passengers, {X_raw.shape[1]} features")
print(f"Survival rate: {y.mean():.1%}")
print(f"\nFeatures: pclass, age, sex(0=M,1=F), sibsp, fare")
```

---

## Step 2 — Feature Engineering

```python
# Add engineered features
age_group = (age < 16).astype(float)      # child flag
alone     = (sibsp == 0).astype(float)    # traveling alone
fare_log  = np.log1p(fare)               # log-fare (skewed distribution)

X = np.column_stack([pclass, age, sex, sibsp, fare_log, age_group, alone])
feature_names = ['pclass', 'age', 'sex', 'sibsp', 'fare_log', 'is_child', 'is_alone']

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {feature_names}")

# Check feature distributions
for i, name in enumerate(feature_names):
    print(f"  {name:<12}: mean={X[:,i].mean():.2f}, std={X[:,i].std():.2f}")
```

---

## Step 3 — Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify preserves class ratio
)

print(f"Train: {len(X_train)} samples, survival rate={y_train.mean():.1%}")
print(f"Test:  {len(X_test)} samples,  survival rate={y_test.mean():.1%}")

# Normalize
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_s  = scaler.transform(X_test)        # transform test with train stats
```

---

## Step 4 — Train and Compare 5 Algorithms

```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)':           SVC(kernel='rbf', C=1.0, probability=True),
}

results = {}

print(f"\n{'Model':<25} {'CV Acc':>8} {'CV Std':>8} {'Test Acc':>10}")
print("-" * 55)

for name, model in models.items():
    # 5-fold cross-validation on training set
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')

    # Final evaluation on test set
    model.fit(X_train_s, y_train)
    test_acc = model.score(X_test_s, y_test)

    results[name] = {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(), 'test_acc': test_acc}
    print(f"{name:<25} {cv_scores.mean():>8.4f} {cv_scores.std():>8.4f} {test_acc:>10.4f}")
```

---

## Step 5 — Deep-Dive Best Model

```python
# Find best model
best_name = max(results, key=lambda k: results[k]['test_acc'])
best_model = models[best_name]

print(f"\nBest model: {best_name}")
print(f"Test accuracy: {results[best_name]['test_acc']:.4f}")

# Full classification report
y_pred = best_model.predict(X_test_s)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# Feature importance (if Random Forest wins)
if hasattr(best_model, 'feature_importances_'):
    fi   = best_model.feature_importances_
    rank = np.argsort(fi)[::-1]
    print(f"Feature Importances:")
    for idx in rank:
        bar = "█" * int(fi[idx] * 50)
        print(f"  {feature_names[idx]:<12}: {fi[idx]:.4f} {bar}")
```

---

## Step 6 — Bias-Variance Analysis

```python
print("\nBias-Variance Analysis:")
print(f"{'Model':<25} {'Train Acc':>10} {'Test Acc':>10} {'Overfit':>10}")
print("-" * 57)

for name, model in models.items():
    train_acc = model.score(X_train_s, y_train)
    test_acc  = model.score(X_test_s, y_test)
    overfit   = train_acc - test_acc
    state     = "High overfit" if overfit > 0.05 else "Good"
    print(f"{name:<25} {train_acc:>10.4f} {test_acc:>10.4f} {overfit:>10.4f}  {state}")
```

---

## Step 7 — Predict New Passenger

```python
# Predict survival probability for specific passengers
test_passengers = [
    {'pclass': 1, 'age': 35, 'sex': 1, 'sibsp': 0, 'fare': 100},  # 1st class woman
    {'pclass': 3, 'age': 25, 'sex': 0, 'sibsp': 0, 'fare': 8},    # 3rd class man
    {'pclass': 2, 'age': 8,  'sex': 0, 'sibsp': 1, 'fare': 25},   # young boy 2nd class
]

print("\nPredictions for new passengers:")
print(f"{'Passenger':<35} {'P(Survive)':>12} {'Prediction':>12}")
print("-" * 61)

for i, p in enumerate(test_passengers):
    fare_log_  = np.log1p(p['fare'])
    age_group_ = float(p['age'] < 16)
    alone_     = float(p['sibsp'] == 0)

    features = np.array([[p['pclass'], p['age'], p['sex'], p['sibsp'],
                          fare_log_, age_group_, alone_]])
    features_s = scaler.transform(features)

    # Use best model with probability
    if hasattr(best_model, 'predict_proba'):
        prob = best_model.predict_proba(features_s)[0, 1]
    else:
        prob = best_model.decision_function(features_s)[0]
        prob = 1 / (1 + np.exp(-prob))

    pred = "SURVIVE" if prob > 0.5 else "DIED"
    desc = f"Class {p['pclass']}, {'F' if p['sex']==1 else 'M'}, age={p['age']}"
    print(f"{desc:<35} {prob:>12.3f} {pred:>12}")
```

---

## What This Project Demonstrated

| Module Concept | Where it appeared |
|---------------|------------------|
| Logistic Regression | Direct implementation, probability output |
| Bias-Variance | Train vs test accuracy gap analysis |
| Cross-validation | 5-fold CV for model selection |
| Decision Tree | Standalone and base for RF/GBM |
| Random Forest | Best model, feature importance |
| Gradient Boosting | Sequential residual correction |
| SVM | Kernel-based classification |
| Feature engineering | Log transform, binary flags |
| Normalization | StandardScaler before SVM/LR |
| Classification metrics | Accuracy, F1, precision, recall |

Every algorithm you learned appeared in one coherent pipeline. This is exactly how a Kaggle competition or production ML pipeline is structured.

---

*Next: [Module 03 — Databases & Vector DBs](03-databases.md)*
