import numpy as np
from sklearn.datasets import make_classification, make_circles, make_moons, load_breast_cancer
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Kernel functions (for intuition) ─────────────────────────

def kernel_linear(X1, X2):
    return X1 @ X2.T


def kernel_rbf(X1, X2, gamma=1.0):
    """K(x, x') = exp(-γ ||x - x'||²)"""
    sq_dists = (np.sum(X1**2, axis=1)[:, None]
                + np.sum(X2**2, axis=1)[None, :]
                - 2 * X1 @ X2.T)
    return np.exp(-gamma * sq_dists)


def kernel_poly(X1, X2, degree=3, gamma=1.0, coef=0.0):
    """K(x, x') = (γ x·x' + r)^d"""
    return (gamma * X1 @ X2.T + coef) ** degree


def kernel_sigmoid(X1, X2, gamma=0.01, coef=0.0):
    """K(x, x') = tanh(γ x·x' + r)"""
    return np.tanh(gamma * X1 @ X2.T + coef)


def kernel_matrix_properties(K, name):
    """Check if kernel matrix is PSD (Mercer condition)."""
    eigenvalues = np.linalg.eigvalsh(K)
    is_psd = np.all(eigenvalues >= -1e-8)
    min_eval = eigenvalues.min()
    print(f"  {name:20s}: PSD={is_psd}, min_eigenvalue={min_eval:.4f}")


# ── SMO-inspired hinge loss gradient ─────────────────────────

class LinearSVMScratch:
    """
    Linear SVM via subgradient descent on hinge loss.
    min_{w,b}  (1/2)||w||² + C * Σ max(0, 1 - y_i(w·x_i + b))
    """

    def __init__(self, C=1.0, lr=0.001, n_iters=1000):
        self.C = C
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        n, d = X.shape
        y_pm = 2 * y - 1    # convert 0/1 → -1/+1
        self.w = np.zeros(d)
        self.b = 0.0

        for t in range(1, self.n_iters + 1):
            scores = y_pm * (X @ self.w + self.b)
            hinge = np.maximum(0, 1 - scores)
            loss = 0.5 * np.dot(self.w, self.w) + self.C * np.mean(hinge)
            self.losses.append(loss)

            # Subgradient: for violated constraints (hinge > 0)
            mask = hinge > 0
            grad_w = self.w - self.C / n * (y_pm[mask][:, None] * X[mask]).sum(axis=0)
            grad_b = -self.C / n * y_pm[mask].sum()

            # Decaying learning rate (1/t schedule)
            lr_t = self.lr / np.sqrt(t)
            self.w -= lr_t * grad_w
            self.b -= lr_t * grad_b

        return self

    def predict(self, X):
        scores = X @ self.w + self.b
        return (scores >= 0).astype(int)

    def decision_function(self, X):
        return X @ self.w + self.b

    @property
    def margin(self):
        """Margin = 2 / ||w||"""
        return 2 / (np.linalg.norm(self.w) + 1e-10)

    def n_support_vectors(self, X, y):
        """Points with hinge loss > 0 (on or inside margin)."""
        y_pm = 2 * y - 1
        scores = y_pm * (X @ self.w + self.b)
        return np.sum(scores <= 1.0)


# ── Main ──────────────────────────────────────────────────────

def main():
    section("1. KERNEL MATRIX PROPERTIES (MERCER'S THEOREM)")
    rng = np.random.default_rng(42)
    X_small = rng.standard_normal((20, 3))
    print("Checking PSD property (Mercer condition) for kernels on 20 random points:")
    kernel_matrix_properties(kernel_linear(X_small, X_small), "Linear")
    kernel_matrix_properties(kernel_rbf(X_small, X_small, gamma=1.0), "RBF (γ=1)")
    kernel_matrix_properties(kernel_rbf(X_small, X_small, gamma=0.1), "RBF (γ=0.1)")
    kernel_matrix_properties(kernel_poly(X_small, X_small, degree=3), "Polynomial (d=3)")
    kernel_matrix_properties(kernel_sigmoid(X_small, X_small, gamma=0.01), "Sigmoid")
    print("  Note: Sigmoid kernel is NOT always PSD → not a valid Mercer kernel in general")

    section("2. LINEAR SVM FROM SCRATCH (SUBGRADIENT DESCENT)")
    X_lin, y_lin = make_classification(n_samples=300, n_features=4, n_informative=3,
                                       n_redundant=0, class_sep=1.5, random_state=0)
    X_lin = StandardScaler().fit_transform(X_lin)
    X_tr, X_te, y_tr, y_te = train_test_split(X_lin, y_lin, test_size=0.25, random_state=0)

    for C in [0.01, 0.1, 1.0, 10.0]:
        svm = LinearSVMScratch(C=C, lr=0.05, n_iters=500).fit(X_tr, y_tr)
        acc = accuracy_score(y_te, svm.predict(X_te))
        sv_count = svm.n_support_vectors(X_tr, y_tr)
        print(f"  C={C:5.2f}: acc={acc:.4f}, ||w||={np.linalg.norm(svm.w):.4f}, "
              f"margin={svm.margin:.4f}, #SV≈{sv_count}")

    section("3. C PARAMETER INTERPRETATION")
    print("C controls regularization strength:")
    print("  Small C → large margin, more misclassifications allowed (high bias, low variance)")
    print("  Large C → small margin, fewer misclassifications (low bias, high variance)")
    sk_lin = LinearSVC(dual=False, max_iter=5000)
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        sk_lin.set_params(C=C)
        sk_lin.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, sk_lin.predict(X_te))
        print(f"  LinearSVC C={C:6.3f}: acc={acc:.4f}, ||w||={np.linalg.norm(sk_lin.coef_):.4f}")

    section("4. KERNEL SVM — NON-LINEAR DATA")
    datasets = {
        "Circles": make_circles(n_samples=300, noise=0.08, factor=0.4, random_state=1),
        "Moons":   make_moons(n_samples=300, noise=0.1, random_state=1),
    }

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    for ds_name, (X_ds, y_ds) in datasets.items():
        scaler = StandardScaler()
        X_ds_s = scaler.fit_transform(X_ds)
        X_dtr, X_dte, y_dtr, y_dte = train_test_split(X_ds_s, y_ds, test_size=0.25, random_state=0)
        print(f"\n  {ds_name} dataset:")
        print(f"  {'Kernel':10s}  {'Test Acc':>10}  {'n_SV':>8}")
        print("  " + "-" * 35)
        for kern in kernels:
            svc = SVC(kernel=kern, C=1.0, gamma="scale").fit(X_dtr, y_dtr)
            acc = accuracy_score(y_dte, svc.predict(X_dte))
            print(f"  {kern:10s}  {acc:>10.4f}  {svc.n_support_vectors_[0] + svc.n_support_vectors_[1]:>8}")

    section("5. RBF KERNEL — γ AND C GRID SEARCH")
    X_rbf, y_rbf = make_classification(n_samples=500, n_features=10, n_informative=6,
                                       random_state=7)
    X_rbf_s = StandardScaler().fit_transform(X_rbf)
    X_rbf_tr, X_rbf_te, y_rbf_tr, y_rbf_te = train_test_split(X_rbf_s, y_rbf, test_size=0.2, random_state=0)

    pipe = Pipeline([("svc", SVC(kernel="rbf", probability=False))])
    param_grid = {"svc__C": [0.1, 1.0, 10.0], "svc__gamma": [0.001, 0.01, 0.1, "scale"]}
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(X_rbf_tr, y_rbf_tr)

    print(f"Best params: {gs.best_params_}")
    print(f"Best CV acc: {gs.best_score_:.4f}")
    print(f"Test acc:    {accuracy_score(y_rbf_te, gs.predict(X_rbf_te)):.4f}")

    section("6. SVM ON BREAST CANCER — FULL PIPELINE")
    bc = load_breast_cancer()
    X_bc, y_bc = bc.data, bc.target
    X_bc_tr, X_bc_te, y_bc_tr, y_bc_te = train_test_split(X_bc, y_bc, test_size=0.2, random_state=0)

    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)),
    ])
    svm_pipe.fit(X_bc_tr, y_bc_tr)
    y_bc_pred = svm_pipe.predict(X_bc_te)

    print(f"Breast Cancer (30 features, binary):")
    print(classification_report(y_bc_te, y_bc_pred,
                                target_names=bc.target_names))

    svc_model = svm_pipe.named_steps["svc"]
    print(f"Number of support vectors: {svc_model.n_support_}")
    print(f"Total SVs: {svc_model.n_support_.sum()} / {len(y_bc_tr)} training points "
          f"({svc_model.n_support_.sum()/len(y_bc_tr)*100:.1f}%)")

    section("7. HINGE LOSS VS CROSS-ENTROPY")
    z = np.linspace(-3, 3, 100)
    y_val = 1
    hinge = np.maximum(0, 1 - y_val * z)
    log_loss = np.log(1 + np.exp(-y_val * z))
    print(f"At z=y*f(x) (higher = more confident correct prediction):")
    print(f"{'z=y*f(x)':>10}  {'Hinge loss':>12}  {'Log loss':>10}")
    print("-" * 38)
    for z_val in [-2, -1, 0, 0.5, 1.0, 1.5, 2.0]:
        h = max(0, 1 - z_val)
        ll = np.log(1 + np.exp(-z_val))
        print(f"{z_val:>10.1f}  {h:>12.4f}  {ll:>10.4f}")
    print("\nHinge loss = 0 for z ≥ 1 (correctly classified with margin)")
    print("Log loss → 0 asymptotically but never exactly 0")


if __name__ == "__main__":
    main()
