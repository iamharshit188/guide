"""
Data drift and concept drift detection — KS test, Chi-squared, PSI, MMD from scratch.
pip install numpy scipy scikit-learn
scipy is optional — KS and chi-squared also implemented from scratch.
"""

import math
import numpy as np
import collections

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ══════════════════════════════════════════════════════════════
# 1. STATISTICAL TESTS — FROM SCRATCH
# ══════════════════════════════════════════════════════════════

def empirical_cdf(data, points):
    """
    Compute empirical CDF F(x) = P(X <= x) at given points.
    Returns array of CDF values same length as points.
    """
    data_sorted = np.sort(data)
    n = len(data_sorted)
    return np.searchsorted(data_sorted, points, side="right") / n


def ks_statistic(ref: np.ndarray, prod: np.ndarray) -> tuple:
    """
    Kolmogorov-Smirnov two-sample test.
    D = sup_x |F_ref(x) - F_prod(x)|

    Returns (D, p_value).
    p-value computed via Kolmogorov distribution approximation.
    """
    combined = np.sort(np.concatenate([ref, prod]))
    cdf_ref  = empirical_cdf(ref,  combined)
    cdf_prod = empirical_cdf(prod, combined)
    D = float(np.abs(cdf_ref - cdf_prod).max())

    # Asymptotic p-value: P(K > D*sqrt(n_eff)) where K ~ Kolmogorov distribution
    n1, n2 = len(ref), len(prod)
    n_eff = math.sqrt(n1 * n2 / (n1 + n2))
    # Kolmogorov CDF: P(K <= z) ≈ 1 - 2*sum_{k=1}^inf (-1)^(k-1) exp(-2k^2 z^2)
    z = D * n_eff
    if z < 1e-10:
        p_val = 1.0
    else:
        s = 0.0
        for k in range(1, 50):
            term = ((-1)**(k-1)) * math.exp(-2 * k**2 * z**2)
            s += term
        p_val = max(0.0, min(1.0, 2 * s))

    return D, p_val


def chi_squared_test(ref_counts: np.ndarray, prod_counts: np.ndarray) -> tuple:
    """
    Chi-squared goodness-of-fit test for categorical distributions.
    ref_counts, prod_counts: integer count arrays of same length.

    Returns (chi2_stat, p_value, dof).
    """
    ref_counts  = np.array(ref_counts, dtype=float)
    prod_counts = np.array(prod_counts, dtype=float)

    # Scale reference to same total as production
    n_ref  = ref_counts.sum()
    n_prod = prod_counts.sum()
    expected = ref_counts * (n_prod / n_ref)

    # chi2 = sum((O - E)^2 / E)  (skip bins where E = 0)
    mask = expected > 0
    chi2 = float(np.sum((prod_counts[mask] - expected[mask])**2 / expected[mask]))
    dof = int(mask.sum()) - 1

    # P-value from chi-squared distribution (upper tail)
    # Using regularized incomplete gamma: P(chi2 > x) = 1 - gamma(k/2, x/2) / Gamma(k/2)
    if SCIPY_AVAILABLE:
        p_val = float(scipy_stats.chi2.sf(chi2, df=dof))
    else:
        # Approximation via standard normal for large dof
        # Wilson-Hilferty: (chi2/dof)^(1/3) ≈ N(1 - 2/(9*dof), 2/(9*dof))
        if dof > 0:
            mu = 1 - 2 / (9 * dof)
            sigma = math.sqrt(2 / (9 * dof))
            z = ((chi2 / dof) ** (1/3) - mu) / sigma
            # P(Z > z) for standard normal
            p_val = max(0.0, 0.5 * math.erfc(z / math.sqrt(2)))
        else:
            p_val = 0.0

    return chi2, p_val, dof


def population_stability_index(ref: np.ndarray, prod: np.ndarray,
                                n_bins: int = 10) -> tuple:
    """
    PSI = sum((A_j - E_j) * ln(A_j / E_j))

    A_j: actual (production) fraction in bin j
    E_j: expected (reference) fraction in bin j

    Returns (psi_value, per_bin details).

    Rule of thumb:
      PSI < 0.1  — no significant change
      PSI < 0.2  — slight change, monitor
      PSI >= 0.2 — significant change, investigate
    """
    bins = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    bins[0]  -= 1e-10   # include minimum
    bins[-1] += 1e-10   # include maximum

    ref_counts  = np.histogram(ref,  bins=bins)[0].astype(float)
    prod_counts = np.histogram(prod, bins=bins)[0].astype(float)

    eps = 1e-8
    ref_frac  = np.maximum(ref_counts  / ref_counts.sum(),  eps)
    prod_frac = np.maximum(prod_counts / prod_counts.sum(), eps)

    bin_psi = (prod_frac - ref_frac) * np.log(prod_frac / ref_frac)
    psi = float(bin_psi.sum())

    details = [
        {"bin": i, "ref_frac": float(ref_frac[i]),
         "prod_frac": float(prod_frac[i]), "bin_psi": float(bin_psi[i])}
        for i in range(n_bins)
    ]
    return psi, details


def maximum_mean_discrepancy(X_ref: np.ndarray, X_prod: np.ndarray,
                              gamma: float = 1.0) -> float:
    """
    MMD^2 with RBF kernel k(x,x') = exp(-gamma * ||x-x'||^2).

    MMD^2 = E[k(X,X')] - 2*E[k(X,Y)] + E[k(Y,Y')]
    where X ~ P_ref, Y ~ P_prod.

    Returns MMD (not squared) — scale-free distance between distributions.
    Unbiased estimator (excludes diagonal terms).
    """

    def rbf_kernel(A, B, gamma):
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
        sq_a = np.sum(A**2, axis=1, keepdims=True)
        sq_b = np.sum(B**2, axis=1, keepdims=True)
        sq_dist = sq_a + sq_b.T - 2 * A @ B.T
        return np.exp(-gamma * sq_dist)

    n = len(X_ref)
    m = len(X_prod)

    K_rr = rbf_kernel(X_ref,  X_ref,  gamma)
    K_pp = rbf_kernel(X_prod, X_prod, gamma)
    K_rp = rbf_kernel(X_ref,  X_prod, gamma)

    # Unbiased: exclude diagonal
    np.fill_diagonal(K_rr, 0)
    np.fill_diagonal(K_pp, 0)

    mmd2 = (K_rr.sum() / (n*(n-1))
           + K_pp.sum() / (m*(m-1))
           - 2 * K_rp.mean())

    return float(math.sqrt(max(mmd2, 0)))


def jensen_shannon_divergence(ref: np.ndarray, prod: np.ndarray,
                               n_bins: int = 20) -> float:
    """
    JS divergence ∈ [0, 1] (base-2 logarithm → bounded by 1).
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = (P+Q)/2.
    """
    bins = np.linspace(
        min(ref.min(), prod.min()) - 1e-8,
        max(ref.max(), prod.max()) + 1e-8,
        n_bins + 1,
    )
    p = np.histogram(ref,  bins=bins)[0].astype(float) + 1e-10
    q = np.histogram(prod, bins=bins)[0].astype(float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.sum(a * np.log2(a / b))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


# ══════════════════════════════════════════════════════════════
# 2. DRIFT DETECTOR CLASS
# ══════════════════════════════════════════════════════════════

class DriftDetector:
    """
    Per-feature drift monitor. Stores reference distribution;
    call detect(production_data) to get a drift report.
    """

    def __init__(self, alpha: float = 0.05, psi_threshold: float = 0.2,
                 mmd_threshold: float = 0.1):
        self.alpha = alpha
        self.psi_threshold = psi_threshold
        self.mmd_threshold = mmd_threshold
        self._reference: dict = {}

    def fit(self, X: np.ndarray, feature_names: list = None):
        """Store reference distribution (training data)."""
        n_features = X.shape[1]
        self._feature_names = (feature_names or
                               [f"feature_{i}" for i in range(n_features)])
        for i, name in enumerate(self._feature_names):
            self._reference[name] = X[:, i].copy()
        self._X_ref = X.copy()
        return self

    def detect(self, X_prod: np.ndarray) -> dict:
        """
        Run all drift tests on production data.
        Returns per-feature results + overall drift flag.
        """
        results = {}
        any_drift = False

        for i, name in enumerate(self._feature_names):
            ref  = self._reference[name]
            prod = X_prod[:, i]

            ks_d, ks_p  = ks_statistic(ref, prod)
            psi, _      = population_stability_index(ref, prod)
            js          = jensen_shannon_divergence(ref, prod)

            ks_drift  = ks_p < self.alpha
            psi_drift = psi >= self.psi_threshold

            drift = ks_drift or psi_drift
            any_drift = any_drift or drift

            results[name] = {
                "ks_statistic":  round(ks_d, 4),
                "ks_p_value":    round(ks_p, 4),
                "ks_drift":      ks_drift,
                "psi":           round(psi, 4),
                "psi_drift":     psi_drift,
                "js_divergence": round(js, 4),
                "drift_detected": drift,
            }

        # Multivariate MMD
        mmd = maximum_mean_discrepancy(self._X_ref, X_prod, gamma=1.0)
        mmd_drift = mmd >= self.mmd_threshold

        return {
            "features": results,
            "mmd": round(mmd, 4),
            "mmd_drift": mmd_drift,
            "overall_drift": any_drift or mmd_drift,
            "n_features_drifted": sum(
                1 for r in results.values() if r["drift_detected"]
            ),
        }


# ══════════════════════════════════════════════════════════════
# 3. SCENARIO GENERATORS
# ══════════════════════════════════════════════════════════════

def generate_no_drift(n=500, d=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


def generate_mean_shift(n=500, d=4, shift=1.0, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:, 0] += shift   # shift only feature 0
    return X


def generate_variance_change(n=500, d=4, scale=3.0, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:, 1] *= scale   # scale only feature 1
    return X


def generate_distribution_change(n=500, d=4, seed=3):
    """Replace feature 2 with bimodal distribution."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    # Bimodal: 50% from N(-2,0.5), 50% from N(+2,0.5)
    half = n // 2
    X[:half, 2]  = rng.normal(-2, 0.5, half)
    X[half:, 2]  = rng.normal(+2, 0.5, n - half)
    return X


def generate_categorical_drift():
    """Reference vs drifted categorical distributions."""
    ref_probs   = [0.5, 0.3, 0.2]          # 3 classes
    drift_probs = [0.2, 0.2, 0.6]          # class 2 surged
    rng = np.random.default_rng(42)
    ref_counts  = (np.array(ref_probs)   * 500).astype(int)
    prod_counts = (np.array(drift_probs) * 500).astype(int)
    return ref_counts, prod_counts


# ══════════════════════════════════════════════════════════════
# 4. PREDICTION DRIFT (CONCEPT DRIFT PROXY)
# ══════════════════════════════════════════════════════════════

def demo_prediction_drift():
    if not SKLEARN_AVAILABLE:
        return

    section("PREDICTION DRIFT — CONCEPT DRIFT PROXY")
    print("""
  When labels are unavailable in production (common), monitor:
    1. Prediction distribution shift (proxy for concept drift).
    2. Model confidence distribution shift.
    3. Feature importance stability.
  Full concept drift detection requires ground-truth labels + time window.
    """)

    data = load_iris()
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_tr, y_tr)

    # Reference predictions
    ref_probs = model.predict_proba(X_te)           # (n_te, 3)
    ref_preds = model.predict(X_te)

    # Simulate production with covariate shift: add noise to features
    rng = np.random.default_rng(7)
    X_shifted = X_te + rng.normal(0, 0.8, X_te.shape)
    prod_probs = model.predict_proba(X_shifted)
    prod_preds = model.predict(X_shifted)

    acc_ref  = accuracy_score(y_te, ref_preds)
    acc_prod = accuracy_score(y_te, prod_preds)

    print(f"  Reference accuracy:   {acc_ref:.4f}")
    print(f"  Production accuracy:  {acc_prod:.4f}  "
          f"(degraded due to covariate shift)")
    print(f"  Accuracy drop:        {acc_ref - acc_prod:.4f}")

    # KS on max confidence (proxy for distribution shift)
    ref_conf  = ref_probs.max(axis=1)
    prod_conf = prod_probs.max(axis=1)
    ks_d, ks_p = ks_statistic(ref_conf, prod_conf)
    print(f"\n  Max confidence distribution KS test:")
    print(f"    D={ks_d:.4f}  p={ks_p:.4f}  "
          f"{'DRIFT DETECTED' if ks_p < 0.05 else 'no drift'}")

    # PSI on prediction class distribution
    ref_class_counts  = np.bincount(ref_preds,  minlength=3)
    prod_class_counts = np.bincount(prod_preds, minlength=3)
    chi2, p_chi, dof = chi_squared_test(ref_class_counts, prod_class_counts)
    print(f"\n  Prediction class distribution Chi-squared test:")
    print(f"    chi2={chi2:.4f}  dof={dof}  p={p_chi:.4f}  "
          f"{'DRIFT DETECTED' if p_chi < 0.05 else 'no drift'}")


# ══════════════════════════════════════════════════════════════
# 5. DEMOS
# ══════════════════════════════════════════════════════════════

def demo_ks_test():
    section("KOLMOGOROV-SMIRNOV TEST")
    rng = np.random.default_rng(0)

    scenarios = [
        ("No drift (same N(0,1))",         rng.standard_normal(500), rng.standard_normal(500)),
        ("Mean shift (N(0,1) vs N(1,1))",  rng.standard_normal(500), rng.standard_normal(500) + 1.0),
        ("Scale change (σ=1 vs σ=3)",      rng.standard_normal(500), rng.standard_normal(500) * 3),
        ("Distribution change (N vs Exp)", rng.standard_normal(500), rng.exponential(1, 500)),
    ]

    print(f"  {'Scenario':40s}  {'D':>7s}  {'p-value':>9s}  {'Drift?':>8s}")
    print("  " + "-" * 70)
    for name, ref, prod in scenarios:
        D, p = ks_statistic(ref, prod)
        drift = "YES" if p < 0.05 else "no"
        print(f"  {name:40s}  {D:>7.4f}  {p:>9.4f}  {drift:>8s}")

    if SCIPY_AVAILABLE:
        print(f"\n  Validation against scipy.stats.ks_2samp:")
        for name, ref, prod in scenarios:
            D_ours, p_ours = ks_statistic(ref, prod)
            D_scipy, p_scipy = scipy_stats.ks_2samp(ref, prod)
            err_D = abs(D_ours - D_scipy)
            err_p = abs(p_ours - p_scipy)
            print(f"  {name[:30]:30s}  ΔD={err_D:.2e}  Δp={err_p:.2e}")


def demo_psi():
    section("POPULATION STABILITY INDEX (PSI)")
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(1000)

    scenarios = [
        ("No drift",          rng.standard_normal(500)),
        ("Small mean shift",  rng.standard_normal(500) + 0.3),
        ("Large mean shift",  rng.standard_normal(500) + 1.5),
        ("Distribution change (uniform)", rng.uniform(-3, 3, 500)),
    ]

    print(f"  {'Scenario':38s}  {'PSI':>7s}  {'Verdict':>10s}")
    print("  " + "-" * 62)
    for name, prod in scenarios:
        psi, _ = population_stability_index(ref, prod, n_bins=10)
        if psi < 0.1:
            verdict = "stable"
        elif psi < 0.2:
            verdict = "slight change"
        else:
            verdict = "MAJOR CHANGE"
        print(f"  {name:38s}  {psi:>7.4f}  {verdict:>10s}")


def demo_chi_squared():
    section("CHI-SQUARED TEST — CATEGORICAL DRIFT")
    ref_counts, prod_counts = generate_categorical_drift()
    chi2, p, dof = chi_squared_test(ref_counts, prod_counts)
    total_ref  = ref_counts.sum()
    total_prod = prod_counts.sum()

    print(f"  Reference distribution (n={total_ref}):")
    for i, c in enumerate(ref_counts):
        print(f"    class {i}: {c} ({c/total_ref:.2%})")
    print(f"  Production distribution (n={total_prod}):")
    for i, c in enumerate(prod_counts):
        print(f"    class {i}: {c} ({c/total_prod:.2%})")
    print(f"\n  Chi-squared statistic: {chi2:.4f}")
    print(f"  Degrees of freedom:    {dof}")
    print(f"  p-value:               {p:.4f}")
    print(f"  Drift detected:        {'YES' if p < 0.05 else 'no'}")


def demo_mmd():
    section("MAXIMUM MEAN DISCREPANCY (MMD)")
    rng = np.random.default_rng(2)
    ref = rng.standard_normal((300, 4))

    scenarios = [
        ("No drift",          rng.standard_normal((300, 4))),
        ("Small shift (0.3)", rng.standard_normal((300, 4)) + 0.3),
        ("Large shift (2.0)", rng.standard_normal((300, 4)) + 2.0),
        ("Scale change ×3",   rng.standard_normal((300, 4)) * 3),
    ]

    print(f"  {'Scenario':30s}  {'MMD':>8s}  {'>0.1?':>8s}")
    print("  " + "-" * 52)
    for name, prod in scenarios:
        mmd = maximum_mean_discrepancy(ref, prod, gamma=1.0)
        flag = "DRIFT" if mmd > 0.1 else "ok"
        print(f"  {name:30s}  {mmd:>8.4f}  {flag:>8s}")


def demo_full_drift_report():
    section("FULL DRIFT REPORT — 4 FEATURES")
    rng = np.random.default_rng(3)
    X_ref = rng.standard_normal((600, 4))
    feature_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]

    # Introduce drift selectively
    X_prod = rng.standard_normal((300, 4))
    X_prod[:, 0] += 1.5          # sepal_len: mean shift
    X_prod[:, 2] *= 2.5          # petal_len: variance change
    # sepal_wid and petal_wid: no drift

    detector = DriftDetector(alpha=0.05, psi_threshold=0.2, mmd_threshold=0.1)
    detector.fit(X_ref, feature_names)
    report = detector.detect(X_prod)

    print(f"  Overall drift detected: {report['overall_drift']}")
    print(f"  Features drifted:       {report['n_features_drifted']} / {len(feature_names)}")
    print(f"  Multivariate MMD:       {report['mmd']} ({'drift' if report['mmd_drift'] else 'ok'})")
    print()
    print(f"  {'Feature':12s}  {'KS-D':>7s}  {'KS-p':>7s}  {'PSI':>7s}  "
          f"{'JS':>7s}  {'KS drift':>9s}  {'PSI drift':>9s}  {'DRIFT':>6s}")
    print("  " + "-" * 80)
    for fname, r in report["features"].items():
        print(f"  {fname:12s}  {r['ks_statistic']:>7.4f}  {r['ks_p_value']:>7.4f}  "
              f"{r['psi']:>7.4f}  {r['js_divergence']:>7.4f}  "
              f"{'YES':>9s if r['ks_drift'] else 'no':>9s}  "
              f"{'YES':>9s if r['psi_drift'] else 'no':>9s}  "
              f"{'DRIFT' if r['drift_detected'] else 'ok':>6s}")


def demo_monitoring_pipeline():
    section("PRODUCTION MONITORING PIPELINE DESIGN")
    print("""
  Architecture:
    Inference service
      → Log (features, prediction, confidence) per request to feature store
        → Batch job (hourly) aggregates window of N=1000 predictions
          → DriftDetector.detect(X_window) vs training reference
            → If overall_drift=True:
                Alert → Slack/PagerDuty
                Trigger retraining pipeline

  Alert thresholds (tune to dataset):
    KS p-value  < 0.01  → definite drift (conservative)
    PSI         > 0.2   → major change
    MMD         > 0.1   → multivariate shift
    Accuracy drop > 5%  → performance regression (needs labels)

  When no labels available:
    Monitor: feature distributions, prediction confidence, prediction class ratios
    Proxy metrics: mean confidence, entropy of prediction distribution, OOD score

  Tools:
    Evidently AI   — open source drift + data quality reports
    Whylogs        — lightweight profile-based drift detection
    Arize          — cloud platform (LLM monitoring included)
    Fiddler        — explainability + drift (enterprise)
    MLflow         — basic metric logging, not dedicated monitoring
    """)


def main():
    demo_ks_test()
    demo_psi()
    demo_chi_squared()
    demo_mmd()
    demo_full_drift_report()
    demo_prediction_drift()
    demo_monitoring_pipeline()


if __name__ == "__main__":
    main()
