import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── K-Fold cross-validation from scratch ─────────────────────

def kfold_split(n, k, seed=42):
    """Yields (train_idx, val_idx) for k folds."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


def cross_val_score(model_fn, X, y, k=5, metric_fn=None):
    """
    model_fn: callable returning a fit sklearn-compatible model
    metric_fn: fn(y_true, y_pred) -> float
    """
    if metric_fn is None:
        metric_fn = lambda yt, yp: np.mean(yt == yp)  # accuracy
    scores = []
    for train_idx, val_idx in kfold_split(len(X), k):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(metric_fn(y[val_idx], preds))
    return np.array(scores)


# ── Bias-Variance tradeoff demo ───────────────────────────────

def bias_variance_demo(X, y, X_test, y_test, n_bootstrap=30, seed=0):
    """
    Decompose bias² + variance + noise for decision trees at varying depths.
    Uses bootstrap sampling: E[(y - f̂(x))²] ≈ Bias² + Variance
    """
    rng = np.random.default_rng(seed)
    depths = [1, 2, 4, 8, 15, None]
    n = len(X)

    print(f"{'Depth':>8}  {'Bias²':>10}  {'Variance':>10}  {'Avg MSE':>10}  {'Status'}")
    print("-" * 55)

    for depth in depths:
        preds_bootstrap = []
        for _ in range(n_bootstrap):
            boot_idx = rng.integers(0, n, size=n)
            tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
            tree.fit(X[boot_idx], y[boot_idx])
            preds_bootstrap.append(tree.predict_proba(X_test)[:, 1])

        preds_array = np.array(preds_bootstrap)  # (n_bootstrap, n_test)
        mean_pred = preds_array.mean(axis=0)

        bias_sq = np.mean((mean_pred - y_test) ** 2)
        variance = np.mean(preds_array.var(axis=0))
        avg_mse = bias_sq + variance

        label = ("Underfitting" if depth == 1
                 else "Overfitting" if depth is None or depth >= 12
                 else "")
        depth_str = str(depth) if depth is not None else "∞"
        print(f"{depth_str:>8}  {bias_sq:>10.4f}  {variance:>10.4f}  {avg_mse:>10.4f}  {label}")


# ── ROC & PR curves (from scratch) ───────────────────────────

def roc_curve_scratch(y_true, y_score):
    thresholds = np.sort(np.unique(y_score))[::-1]
    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    tprs, fprs = [0.0], [0.0]
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tprs.append(tp / pos)
        fprs.append(fp / neg)
    tprs.append(1.0); fprs.append(1.0)
    return np.array(fprs), np.array(tprs)


def auc_trapezoidal(x, y):
    return float(np.trapz(y, x))


def pr_curve_scratch(y_true, y_score):
    thresholds = np.sort(np.unique(y_score))[::-1]
    precisions, recalls = [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp + 1e-10)
        rec  = tp / (tp + fn + 1e-10)
        precisions.append(prec)
        recalls.append(rec)
    return np.array(recalls), np.array(precisions)


# ── Calibration check ─────────────────────────────────────────

def calibration_check(y_true, y_prob, n_bins=10):
    """
    Reliability diagram data: avg predicted prob vs fraction positive in each bin.
    A well-calibrated model: predicted P ≈ actual fraction.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    print(f"\n{'Bin':>12}  {'Avg Pred':>10}  {'Actual Frac':>12}  {'Count':>7}  {'Calibrated?'}")
    print("-" * 60)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo)
        if mask.sum() == 0:
            continue
        avg_pred = y_prob[mask].mean()
        actual_frac = y_true[mask].mean()
        ok = abs(avg_pred - actual_frac) < 0.1
        print(f"[{lo:.1f},{hi:.1f}]  {avg_pred:>10.3f}  {actual_frac:>12.3f}  "
              f"{mask.sum():>7}  {'✓' if ok else '✗'}")


# ── Main ──────────────────────────────────────────────────────

def main():
    section("1. K-FOLD CROSS-VALIDATION FROM SCRATCH")

    X, y = make_classification(n_samples=500, n_features=10, n_informative=6,
                               n_redundant=2, random_state=42)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    for k in [3, 5, 10]:
        scores = cross_val_score(lambda: LogisticRegression(max_iter=200), X_s, y, k=k)
        print(f"  k={k:2d}: mean={scores.mean():.4f}  std={scores.std():.4f}  "
              f"scores={scores.round(4)}")

    section("2. BIAS-VARIANCE DECOMPOSITION")
    X_bv, y_bv = make_classification(n_samples=800, n_features=8, n_informative=5,
                                     random_state=0, class_sep=0.8)
    scaler2 = StandardScaler()
    X_bv_s = scaler2.fit_transform(X_bv)
    X_bv_tr, X_bv_te = X_bv_s[:600], X_bv_s[600:]
    y_bv_tr, y_bv_te = y_bv[:600], y_bv[600:]
    bias_variance_demo(X_bv_tr, y_bv_tr, X_bv_te, y_bv_te.astype(float), n_bootstrap=25)

    section("3. ROC CURVE & AUC")
    X_roc, y_roc = make_classification(n_samples=600, n_features=8, n_informative=5,
                                       random_state=7)
    scaler3 = StandardScaler()
    X_roc_s = scaler3.fit_transform(X_roc)
    split = 480
    X_tr3, X_te3 = X_roc_s[:split], X_roc_s[split:]
    y_tr3, y_te3 = y_roc[:split], y_roc[split:]

    model_lr = LogisticRegression(max_iter=200).fit(X_tr3, y_tr3)
    model_tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr3, y_tr3)

    for name, m in [("LogReg", model_lr), ("Tree", model_tree)]:
        proba = m.predict_proba(X_te3)[:, 1]
        fpr, tpr = roc_curve_scratch(y_te3, proba)
        auc_scratch = auc_trapezoidal(fpr, tpr)
        auc_sk = roc_auc_score(y_te3, proba)
        rec, prec = pr_curve_scratch(y_te3, proba)
        pr_auc = auc_trapezoidal(rec, prec)

        print(f"\n{name}:")
        print(f"  AUC-ROC (scratch): {auc_scratch:.4f}")
        print(f"  AUC-ROC (sklearn): {auc_sk:.4f}")
        print(f"  AUC-PR:            {pr_auc:.4f}")

    section("4. CLASSIFICATION REPORT")
    y_pred3 = model_lr.predict(X_te3)
    print(classification_report(y_te3, y_pred3, target_names=["neg", "pos"]))
    cm = confusion_matrix(y_te3, y_pred3)
    tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    print(f"Confusion:\n{cm}")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Sensitivity (Recall) = TP/(TP+FN) = {tp/(tp+fn):.4f}")
    print(f"  Specificity = TN/(TN+FP) = {tn/(tn+fp):.4f}")
    print(f"  Positive Predictive Value = TP/(TP+FP) = {tp/(tp+fp):.4f}")

    section("5. CALIBRATION CHECK")
    proba_lr = model_lr.predict_proba(X_te3)[:, 1]
    calibration_check(y_te3, proba_lr)

    section("6. LEARNING CURVES (TRAINING SIZE EFFECT)")
    print(f"{'Train N':>8}  {'Train Acc':>10}  {'Val Acc':>10}  {'Gap':>8}")
    print("-" * 40)
    X_lc, y_lc = make_classification(n_samples=1000, n_features=10, n_informative=6,
                                     random_state=1)
    X_lc_s = StandardScaler().fit_transform(X_lc)
    X_lc_val, y_lc_val = X_lc_s[800:], y_lc[800:]

    for n_train in [20, 50, 100, 200, 400, 800]:
        X_sub, y_sub = X_lc_s[:n_train], y_lc[:n_train]
        m = LogisticRegression(max_iter=300, C=1.0).fit(X_sub, y_sub)
        tr_acc = np.mean(m.predict(X_sub) == y_sub)
        va_acc = np.mean(m.predict(X_lc_val) == y_lc_val)
        print(f"{n_train:>8}  {tr_acc:>10.4f}  {va_acc:>10.4f}  {tr_acc-va_acc:>8.4f}")

    section("7. CLASS IMBALANCE")
    X_imb, y_imb = make_classification(n_samples=1000, n_features=8, weights=[0.9, 0.1],
                                       n_informative=4, random_state=5)
    X_imb_s = StandardScaler().fit_transform(X_imb)
    split_i = 800
    X_imb_tr, X_imb_te = X_imb_s[:split_i], X_imb_s[split_i:]
    y_imb_tr, y_imb_te = y_imb[:split_i], y_imb[split_i:]
    print(f"Class distribution: neg={np.sum(y_imb_te==0)}, pos={np.sum(y_imb_te==1)}")

    for balance in [None, "balanced"]:
        m = LogisticRegression(class_weight=balance, max_iter=300).fit(X_imb_tr, y_imb_tr)
        y_p = m.predict(X_imb_te)
        proba_p = m.predict_proba(X_imb_te)[:, 1]
        tp = ((y_p==1) & (y_imb_te==1)).sum()
        fp = ((y_p==1) & (y_imb_te==0)).sum()
        fn = ((y_p==0) & (y_imb_te==1)).sum()
        prec = tp / (tp+fp+1e-10)
        rec  = tp / (tp+fn+1e-10)
        f1   = 2*prec*rec/(prec+rec+1e-10)
        auc_i = roc_auc_score(y_imb_te, proba_p)
        lbl = "Default" if balance is None else "Balanced"
        print(f"  {lbl:10s}: Acc={np.mean(y_p==y_imb_te):.3f}  "
              f"Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  AUC={auc_i:.3f}")


if __name__ == "__main__":
    main()
