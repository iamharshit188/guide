import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Decision Tree internals demo ──────────────────────────────

def gini(y):
    """Gini impurity: 1 - Σ p_k²"""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def entropy(y):
    """Shannon entropy: -Σ p_k log2(p_k)"""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y[y >= 0])
    probs = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs))


def information_gain(y, y_left, y_right, criterion="gini"):
    fn = gini if criterion == "gini" else entropy
    n, nl, nr = len(y), len(y_left), len(y_right)
    return fn(y) - (nl / n) * fn(y_left) - (nr / n) * fn(y_right)


def best_split(X, y, n_features=None):
    """
    Find best (feature, threshold) split by information gain.
    n_features: subsample features (Random Forest trick).
    """
    n, d = X.shape
    best_gain, best_feat, best_thresh = -np.inf, None, None
    feature_indices = (np.random.choice(d, n_features, replace=False)
                       if n_features is not None else np.arange(d))

    for j in feature_indices:
        thresholds = np.unique(X[:, j])
        for t in thresholds[:-1]:
            left = y[X[:, j] <= t]
            right = y[X[:, j] > t]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = information_gain(y, left, right)
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, j, t

    return best_feat, best_thresh, best_gain


# ── Minimal Decision Tree ─────────────────────────────────────

class Node:
    __slots__ = ["feature", "threshold", "left", "right", "value"]

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf prediction


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        n_classes = len(np.unique(y))
        # Leaf conditions
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                n_classes == 1):
            return Node(value=np.bincount(y).argmax())

        feat, thresh, gain = best_split(X, y, self.n_features)
        if feat is None or gain <= 0:
            return Node(value=np.bincount(y).argmax())

        left_mask = X[:, feat] <= thresh
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=feat, threshold=thresh, left=left, right=right)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


# ── Minimal Random Forest (illustrative) ─────────────────────

class RandomForest:
    """Illustrative RF: bagging + feature subsampling."""

    def __init__(self, n_estimators=50, max_depth=8, max_features="sqrt", seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed
        self.trees_ = []
        self.oob_score_ = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        n_feat = int(np.sqrt(d)) if self.max_features == "sqrt" else d
        oob_votes = np.full((n, int(y.max()) + 1), 0)

        for _ in range(self.n_estimators):
            boot_idx = rng.integers(0, n, n)
            oob_idx = np.setdiff1d(np.arange(n), boot_idx)
            tree = DecisionTree(max_depth=self.max_depth, n_features=n_feat)
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees_.append((tree, oob_idx))

            if len(oob_idx) > 0:
                preds = tree.predict(X[oob_idx])
                for i, pred in zip(oob_idx, preds):
                    oob_votes[i, pred] += 1

        # OOB score
        oob_preds = oob_votes.argmax(axis=1)
        voted_mask = oob_votes.sum(axis=1) > 0
        self.oob_score_ = np.mean(oob_preds[voted_mask] == y[voted_mask])
        return self

    def predict(self, X):
        votes = np.array([t.predict(X) for t, _ in self.trees_])  # (M, N)
        return np.apply_along_axis(
            lambda col: np.bincount(col).argmax(), axis=0, arr=votes
        )


# ── Main ──────────────────────────────────────────────────────

def main():
    section("1. GINI VS ENTROPY SPLITTING DEMO")
    y_examples = [
        ("Pure node", np.array([0, 0, 0, 0, 0])),
        ("50/50 split (binary)", np.array([0, 0, 1, 1])),
        ("Skewed (binary)", np.array([0, 0, 0, 1])),
        ("3-class uniform", np.array([0, 1, 2])),
    ]
    print(f"{'Case':30s}  {'Gini':>8}  {'Entropy':>10}")
    print("-" * 55)
    for name, y_ex in y_examples:
        print(f"{name:30s}  {gini(y_ex):>8.4f}  {entropy(y_ex):>10.4f}")

    section("2. SINGLE DECISION TREE FROM SCRATCH")
    X, y = make_classification(n_samples=400, n_features=6, n_informative=4,
                               n_redundant=1, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

    for depth in [2, 4, 8, None]:
        tree = DecisionTreeClassifier(max_depth=depth, criterion="gini", random_state=0)
        tree.fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, tree.predict(X_tr))
        te_acc = accuracy_score(y_te, tree.predict(X_te))
        depth_str = str(depth) if depth else "∞"
        print(f"max_depth={depth_str:3s}: train_acc={tr_acc:.4f}, test_acc={te_acc:.4f}  "
              f"({'overfit' if tr_acc - te_acc > 0.08 else 'ok'})")

    section("3. BAGGING — VARIANCE REDUCTION")
    base_acc = []
    n_bootstrap = 20
    rng = np.random.default_rng(0)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(X_tr), len(X_tr))
        t = DecisionTreeClassifier(max_depth=None, random_state=0).fit(X_tr[idx], y_tr[idx])
        base_acc.append(accuracy_score(y_te, t.predict(X_te)))

    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=None),
        n_estimators=50, max_samples=1.0, max_features=1.0,
        random_state=0
    ).fit(X_tr, y_tr)

    print(f"Single tree (avg over {n_bootstrap} bootstraps):")
    print(f"  Mean acc: {np.mean(base_acc):.4f}  Std: {np.std(base_acc):.4f}")
    print(f"Bagging (50 trees, n_init=1):")
    print(f"  Acc: {accuracy_score(y_te, bag.predict(X_te)):.4f}  (variance reduced by ensemble)")

    section("4. RANDOM FOREST (FROM SCRATCH)")
    rf_scratch = RandomForest(n_estimators=30, max_depth=6, seed=42).fit(X_tr, y_tr)
    scratch_acc = accuracy_score(y_te, rf_scratch.predict(X_te))
    print(f"Scratch RF (30 trees, max_depth=6):")
    print(f"  Test acc:   {scratch_acc:.4f}")
    print(f"  OOB score:  {rf_scratch.oob_score_:.4f}")

    section("5. SKLEARN RANDOM FOREST — TUNING")
    print(f"{'n_estimators':>14}  {'max_depth':>10}  {'Test Acc':>10}  {'OOB Acc':>10}")
    print("-" * 50)
    for n_est in [10, 50, 100, 200]:
        for depth in [5, None]:
            rf = RandomForestClassifier(
                n_estimators=n_est, max_depth=depth,
                max_features="sqrt", oob_score=True, random_state=0
            ).fit(X_tr, y_tr)
            te_acc = accuracy_score(y_te, rf.predict(X_te))
            depth_str = str(depth) if depth else "∞"
            print(f"{n_est:>14}  {depth_str:>10}  {te_acc:>10.4f}  {rf.oob_score_:>10.4f}")

    section("6. FEATURE IMPORTANCE — BREAST CANCER DATASET")
    bc = load_breast_cancer()
    X_bc = bc.data
    y_bc = bc.target
    X_bc_tr, X_bc_te, y_bc_tr, y_bc_te = train_test_split(X_bc, y_bc, test_size=0.2, random_state=0)

    rf_bc = RandomForestClassifier(n_estimators=200, max_depth=None, oob_score=True,
                                   random_state=42).fit(X_bc_tr, y_bc_tr)

    print(f"Breast cancer dataset ({X_bc.shape[1]} features):")
    print(f"  Test accuracy: {accuracy_score(y_bc_te, rf_bc.predict(X_bc_te)):.4f}")
    print(f"  OOB score:     {rf_bc.oob_score_:.4f}")

    # MDI importance
    mdi_imp = rf_bc.feature_importances_
    top_mdi_idx = np.argsort(mdi_imp)[::-1][:8]
    print(f"\nTop 8 features by MDI importance:")
    print(f"  {'Feature':35s}  {'MDI Importance':>15}")
    print("  " + "-" * 55)
    for i in top_mdi_idx:
        print(f"  {bc.feature_names[i]:35s}  {mdi_imp[i]:>15.4f}")

    # Permutation importance
    perm = permutation_importance(rf_bc, X_bc_te, y_bc_te, n_repeats=10, random_state=0)
    top_perm_idx = np.argsort(perm.importances_mean)[::-1][:5]
    print(f"\nTop 5 by permutation importance (test set):")
    print(f"  {'Feature':35s}  {'Mean Drop':>10}  {'Std':>8}")
    print("  " + "-" * 58)
    for i in top_perm_idx:
        print(f"  {bc.feature_names[i]:35s}  {perm.importances_mean[i]:>10.4f}  "
              f"{perm.importances_std[i]:>8.4f}")

    section("7. OOB ERROR CONVERGENCE")
    print("OOB error vs n_estimators (convergence):")
    print(f"{'n_trees':>8}  {'OOB Acc':>10}  {'Test Acc':>10}")
    print("-" * 35)
    for n_est in [5, 10, 25, 50, 100, 200, 500]:
        rf_n = RandomForestClassifier(n_estimators=n_est, oob_score=True,
                                      random_state=0, n_jobs=-1).fit(X_bc_tr, y_bc_tr)
        te_a = accuracy_score(y_bc_te, rf_n.predict(X_bc_te))
        print(f"{n_est:>8}  {rf_n.oob_score_:>10.4f}  {te_a:>10.4f}")


if __name__ == "__main__":
    main()
