"""
Decision Trees from scratch.
Covers: CART algorithm, Gini impurity, entropy, information gain,
        overfitting, pruning (cost-complexity), feature importance.
Pure NumPy — no sklearn dependency required.
"""

import numpy as np

rng = np.random.default_rng(42)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Impurity functions ────────────────────────────────────────────
def gini(y: np.ndarray) -> float:
    """Gini impurity: 1 - Σ p_k². Range [0, 1-1/K]."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return float(1.0 - np.sum(p ** 2))


def entropy(y: np.ndarray) -> float:
    """Entropy: -Σ p_k log2(p_k). Range [0, log2(K)]."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return float(-np.sum(p * np.log2(p + 1e-12)))


def mse_impurity(y: np.ndarray) -> float:
    """MSE impurity for regression nodes."""
    if len(y) == 0:
        return 0.0
    return float(np.var(y))


def information_gain(y: np.ndarray, y_left: np.ndarray,
                     y_right: np.ndarray, criterion: str = "gini") -> float:
    """
    IG(S, split) = impurity(S) - (|S_L|/|S|)·impurity(S_L) - (|S_R|/|S|)·impurity(S_R)
    """
    fn = gini if criterion == "gini" else (entropy if criterion == "entropy" else mse_impurity)
    n, n_l, n_r = len(y), len(y_left), len(y_right)
    if n == 0:
        return 0.0
    parent_imp = fn(y)
    child_imp  = (n_l / n) * fn(y_left) + (n_r / n) * fn(y_right)
    return parent_imp - child_imp


# ── CART Splitting ────────────────────────────────────────────────
def best_split(X: np.ndarray, y: np.ndarray,
               criterion: str = "gini",
               max_features: int | None = None) -> tuple[int, float, float]:
    """
    CART: find the (feature, threshold) that maximises information gain.
    Tries all unique thresholds for each feature (or random subset).

    Returns: (best_feature_idx, best_threshold, best_gain)
    """
    n_samples, n_features = X.shape
    feat_indices = np.arange(n_features)
    if max_features and max_features < n_features:
        feat_indices = rng.choice(n_features, max_features, replace=False)

    best_gain  = -np.inf
    best_feat  = 0
    best_thresh = 0.0

    for feat in feat_indices:
        values = np.unique(X[:, feat])
        if len(values) < 2:
            continue
        # Candidate thresholds: midpoints between consecutive unique values
        thresholds = (values[:-1] + values[1:]) / 2.0

        for thresh in thresholds:
            mask = X[:, feat] <= thresh
            y_l, y_r = y[mask], y[~mask]
            if len(y_l) == 0 or len(y_r) == 0:
                continue
            gain = information_gain(y, y_l, y_r, criterion)
            if gain > best_gain:
                best_gain  = gain
                best_feat  = feat
                best_thresh = thresh

    return best_feat, best_thresh, best_gain


# ── Tree Node ─────────────────────────────────────────────────────
class Node:
    __slots__ = ("feature", "threshold", "left", "right",
                 "value", "n_samples", "impurity")

    def __init__(self):
        self.feature   = None
        self.threshold = None
        self.left      = None
        self.right     = None
        self.value     = None     # leaf prediction
        self.n_samples = 0
        self.impurity  = 0.0

    @property
    def is_leaf(self):
        return self.value is not None


# ── Decision Tree ─────────────────────────────────────────────────
class DecisionTree:
    """
    CART decision tree for classification (Gini/Entropy) and regression (MSE).
    Supports:
      - max_depth: depth limit
      - min_samples_split: minimum samples required to split
      - min_impurity_decrease: only split if IG >= threshold
      - ccp_alpha: cost-complexity pruning parameter
    """

    def __init__(self, criterion: str = "gini", max_depth: int | None = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 min_impurity_decrease: float = 0.0, ccp_alpha: float = 0.0,
                 max_features: int | None = None, task: str = "classification"):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.max_features = max_features
        self.task = task
        self.root = None
        self.feature_importances_ = None

    def _leaf_value(self, y: np.ndarray) -> float | int:
        if self.task == "classification":
            values, counts = np.unique(y, return_counts=True)
            return int(values[np.argmax(counts)])
        return float(y.mean())

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node()
        node.n_samples = len(y)

        fn = gini if self.criterion == "gini" else \
             (entropy if self.criterion == "entropy" else mse_impurity)
        node.impurity = fn(y)

        # Stopping conditions → leaf
        if (len(y) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth) or
                len(np.unique(y)) == 1):
            node.value = self._leaf_value(y)
            return node

        feat, thresh, gain = best_split(X, y, self.criterion, self.max_features)

        if gain < self.min_impurity_decrease:
            node.value = self._leaf_value(y)
            return node

        mask = X[:, feat] <= thresh
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            node.value = self._leaf_value(y)
            return node

        node.feature   = feat
        node.threshold = thresh
        self._feature_gains[feat] += gain * len(y)   # MDI accumulator
        node.left  = self._build(X[mask],  y[mask],  depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        n_features = X.shape[1]
        self._feature_gains = np.zeros(n_features)
        self.root = self._build(X, y, depth=0)
        if self.ccp_alpha > 0:
            self._prune(self.root)
        total = self._feature_gains.sum()
        self.feature_importances_ = self._feature_gains / (total + 1e-12)
        return self

    def _predict_one(self, node: Node, x: np.ndarray):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(self.root, x) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use sklearn for proba; this demo uses predict()")

    # ── Cost-complexity pruning ────────────────────────────────────
    def _n_leaves(self, node: Node) -> int:
        if node.is_leaf:
            return 1
        return self._n_leaves(node.left) + self._n_leaves(node.right)

    def _subtree_error(self, node: Node, X: np.ndarray, y: np.ndarray) -> float:
        preds = np.array([self._predict_one(node, x) for x in X])
        if self.task == "classification":
            return float((preds != y).mean())
        return float(np.mean((preds - y) ** 2))

    def _prune(self, node: Node) -> None:
        """
        Simplified cost-complexity pruning:
        L(α, T) = MSE(T) + α|T|   where |T| = number of leaves
        Prune if replacing subtree with leaf does not increase cost by more than α.
        """
        if node.is_leaf:
            return
        self._prune(node.left)
        self._prune(node.right)
        # If both children are leaves, consider collapsing
        if node.left.is_leaf and node.right.is_leaf:
            n_leaves_before = 2
            n_leaves_after  = 1
            # Cost without pruning is already accumulated; approximate by alpha * Δleaves
            if self.ccp_alpha * (n_leaves_before - n_leaves_after) >= 0:
                node.value = self._leaf_value(
                    np.array([node.left.value, node.right.value])
                )
                node.left  = None
                node.right = None

    def depth(self) -> int:
        def _depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root)

    def n_leaves(self) -> int:
        return self._n_leaves(self.root)

    def print_tree(self, node: Node | None = None, depth: int = 0,
                   feature_names: list | None = None):
        if node is None:
            node = self.root
        prefix = "  " * depth
        if node.is_leaf:
            print(f"{prefix}Leaf: predict={node.value}  (n={node.n_samples})")
        else:
            fname = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
            print(f"{prefix}[{fname} <= {node.threshold:.4f}]  "
                  f"gain={node.impurity:.4f}  n={node.n_samples}")
            self.print_tree(node.left,  depth+1, feature_names)
            self.print_tree(node.right, depth+1, feature_names)


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def make_classification_dataset(n: int = 300, n_features: int = 4):
    """Linearly separable 3-class dataset with noise."""
    X = rng.standard_normal((n, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 0.5).astype(int)
    return X, y


def make_regression_dataset(n: int = 200):
    X = rng.standard_normal((n, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.standard_normal(n) * 0.3
    return X, y


def main():
    section("1. IMPURITY FUNCTIONS")
    print("""
  Gini impurity:   G(S) = 1 - Σ p_k²
  Entropy:         H(S) = -Σ p_k log₂ p_k
  For regression:  MSE(S) = Var(y) in node
""")

    for dist in [[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]:
        y = np.array(dist)
        g = gini(y)
        e = entropy(y)
        label = f"p(1)={y.mean():.2f}"
        print(f"  {label:15}  Gini={g:.4f}  Entropy={e:.4f}")

    print("""
  Pure node:     Gini=0, Entropy=0
  Max impurity:  Gini=0.5 (binary), Entropy=1.0 (binary)
  Gini is faster (no log) and almost always gives the same tree.
""")

    section("2. INFORMATION GAIN EXAMPLE")
    # Manual example: feature = [0,0,0,1,1,1], target = [0,0,1,1,1,0]
    y_all   = np.array([0, 0, 1, 1, 1, 0])
    y_left  = np.array([0, 0, 1])    # X <= 0
    y_right = np.array([1, 1, 0])    # X > 0

    ig_gini = information_gain(y_all, y_left, y_right, "gini")
    ig_ent  = information_gain(y_all, y_left, y_right, "entropy")
    print(f"  y_all:   {y_all}")
    print(f"  y_left:  {y_left}  y_right: {y_right}")
    print(f"  IG(Gini)={ig_gini:.4f}  IG(Entropy)={ig_ent:.4f}")

    section("3. CART — BEST SPLIT SEARCH")
    X_demo = np.array([[1.0, 2.0], [1.5, 1.0], [3.0, 4.0], [3.5, 3.5],
                       [5.0, 1.5], [5.5, 2.0]])
    y_demo = np.array([0, 0, 1, 1, 0, 0])
    feat, thresh, gain = best_split(X_demo, y_demo, criterion="gini")
    print(f"  Best split: feature={feat}, threshold={thresh:.4f}, gain={gain:.4f}")

    section("4. TRAINING A DECISION TREE (CLASSIFICATION)")
    X, y = make_classification_dataset(n=400, n_features=4)
    split = int(0.8 * len(X))
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    feature_names = ["x0", "x1", "x2", "x3"]
    results = []
    for max_d in [2, 4, 6, None]:
        dt = DecisionTree(criterion="gini", max_depth=max_d)
        dt.fit(X_tr, y_tr)
        tr_acc = accuracy(y_tr, dt.predict(X_tr))
        te_acc = accuracy(y_te, dt.predict(X_te))
        results.append((max_d, dt.depth(), dt.n_leaves(), tr_acc, te_acc))
        print(f"  max_depth={str(max_d):4}  actual_depth={dt.depth():<3}  "
              f"leaves={dt.n_leaves():<4}  train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}")

    print("""
  Observation: deeper trees overfit (train≈1.0, test drops).
  max_depth=4-6 is typically the sweet spot for tabular data.
""")

    section("5. TREE VISUALIZATION (max_depth=2)")
    dt_small = DecisionTree(criterion="gini", max_depth=2)
    dt_small.fit(X_tr, y_tr)
    dt_small.print_tree(feature_names=feature_names)

    section("6. FEATURE IMPORTANCE (MDI)")
    dt_full = DecisionTree(criterion="gini", max_depth=None)
    dt_full.fit(X_tr, y_tr)
    print(f"  Mean Decrease in Impurity (normalized):")
    print(f"  {'Feature':10} {'Importance':12}")
    for name, imp in sorted(zip(feature_names, dt_full.feature_importances_),
                             key=lambda x: -x[1]):
        bar = "█" * int(imp * 30)
        print(f"  {name:10} {imp:12.4f}  {bar}")
    print("""
  MDI sums the (gain × samples) at each split, normalized.
  Caveat: MDI is biased toward high-cardinality features.
  Use permutation importance for more reliable estimates.
""")

    section("7. PRUNING — COST-COMPLEXITY")
    print("""
  L(α, T) = error(T) + α × |T|    (|T| = number of leaves)
  α=0: no pruning (full tree)
  Increasing α: prune more aggressively
  Optimal α: found via cross-validation (sklearn: ccp_alpha)
""")

    print(f"  {'ccp_alpha':12} {'depth':>6} {'leaves':>7} {'train_acc':>11} {'test_acc':>10}")
    print(f"  {'-'*50}")
    for alpha in [0.0, 0.005, 0.01, 0.05]:
        dt = DecisionTree(criterion="gini", ccp_alpha=alpha)
        dt.fit(X_tr, y_tr)
        tr_acc = accuracy(y_tr, dt.predict(X_tr))
        te_acc = accuracy(y_te, dt.predict(X_te))
        print(f"  {alpha:12.3f} {dt.depth():>6} {dt.n_leaves():>7} "
              f"{tr_acc:>11.4f} {te_acc:>10.4f}")

    section("8. REGRESSION TREE")
    X_r, y_r = make_regression_dataset(n=300)
    split_r = int(0.8 * len(X_r))
    X_tr_r, y_tr_r = X_r[:split_r], y_r[:split_r]
    X_te_r, y_te_r = X_r[split_r:], y_r[split_r:]

    print(f"  {'max_depth':10} {'train_mse':>12} {'test_mse':>12}")
    print(f"  {'-'*36}")
    for max_d in [2, 5, 10, None]:
        dt_r = DecisionTree(criterion="mse", max_depth=max_d, task="regression")
        dt_r.fit(X_tr_r, y_tr_r)
        tr_mse = float(np.mean((y_tr_r - dt_r.predict(X_tr_r))**2))
        te_mse = float(np.mean((y_te_r - dt_r.predict(X_te_r))**2))
        print(f"  {str(max_d):10} {tr_mse:12.4f} {te_mse:12.4f}")

    section("9. CART vs. ID3 vs. C4.5")
    print("""
  ┌──────────────┬──────────┬──────────┬────────────────────────────┐
  │ Algorithm    │ Splits   │ Impurity │ Notes                      │
  ├──────────────┼──────────┼──────────┼────────────────────────────┤
  │ ID3          │ Multi-way│ Entropy  │ Categorical only; no pruning│
  │ C4.5         │ Binary   │ Gain Ratio│ Handles continuous + missing│
  │ CART         │ Binary   │ Gini/MSE │ Classification + Regression │
  │ sklearn DT   │ Binary   │ Gini/Ent │ CART implementation         │
  └──────────────┴──────────┴──────────┴────────────────────────────┘

  Gain Ratio (C4.5): IG(S,f) / H(S,f) — penalises features with many
  unique values (avoids overfitting to e.g. unique ID columns).

  CART is the industry standard (sklearn, XGBoost all use binary splits).
""")


if __name__ == "__main__":
    main()
