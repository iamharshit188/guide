import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.decomposition import PCA as SKLearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── PCA from scratch ──────────────────────────────────────────

class PCA:
    """
    PCA via covariance matrix eigendecomposition.
    Equivalent to SVD of centered data.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None        # (k, d) — rows = principal components
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Covariance matrix: (d, d)
        cov = X_centered.T @ X_centered / (n - 1)

        # Eigendecomposition (returns unsorted)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # eigh for symmetric

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]  # columns = eigenvectors

        k = self.n_components if self.n_components is not None else d
        self.explained_variance_ = eigenvalues[:k]
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[:k] / total_var
        self.components_ = eigenvectors[:, :k].T   # (k, d) — rows = PCs
        self._total_var = total_var
        self._all_eigenvalues = eigenvalues
        return self

    def transform(self, X):
        """Project X onto principal components: Z = (X - μ) @ V_k"""
        return (X - self.mean_) @ self.components_.T  # (N, k)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        """Reconstruct: X̃ = Z @ V_k + μ"""
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X):
        Z = self.transform(X)
        X_rec = self.inverse_transform(Z)
        return np.mean((X - X_rec) ** 2)

    def n_components_for_variance(self, threshold=0.95):
        cumulative = np.cumsum(self._all_eigenvalues) / self._total_var
        return int(np.searchsorted(cumulative, threshold)) + 1


# ── PCA via SVD (equivalent, numerically more stable) ─────────

def pca_svd(X, k):
    """
    Alternative PCA implementation via SVD.
    X_centered = U Σ V^T → PCs = rows of V^T, scores = U Σ
    """
    X_centered = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:k]                    # (k, d)
    scores = X_centered @ components.T     # (N, k)
    eigenvalues = s[:k]**2 / (len(X) - 1)
    return scores, components, eigenvalues


# ── Main ──────────────────────────────────────────────────────

def main():
    section("1. PCA FROM SCRATCH — SIMPLE 2D → 1D")
    rng = np.random.default_rng(42)
    cov_true = np.array([[3.0, 2.0], [2.0, 2.0]])
    X_2d = rng.multivariate_normal([0, 0], cov_true, 300)

    pca_2d = PCA(n_components=2).fit(X_2d)
    print(f"True covariance:\n{cov_true}")
    print(f"\nPCA eigenvalues (≈ variances): {pca_2d.explained_variance_.round(4)}")
    print(f"Sum eigenvalues ≈ tr(Σ) = {np.trace(cov_true)} (actual: {pca_2d.explained_variance_.sum():.4f})")
    print(f"Explained variance ratio: {pca_2d.explained_variance_ratio_.round(4)}")
    print(f"PC1 direction: {pca_2d.components_[0].round(4)}")
    print(f"PC2 direction: {pca_2d.components_[1].round(4)}")
    print(f"PC1 · PC2 = {np.dot(pca_2d.components_[0], pca_2d.components_[1]):.10f}  (orthogonal)")

    # Projection
    Z_2d = pca_2d.transform(X_2d)
    print(f"\nProjected data shape: {Z_2d.shape}")
    print(f"Var(PC1 scores): {Z_2d[:,0].var():.4f}  (≈ λ1={pca_2d.explained_variance_[0]:.4f})")
    print(f"Var(PC2 scores): {Z_2d[:,1].var():.4f}  (≈ λ2={pca_2d.explained_variance_[1]:.4f})")
    print(f"Cov(PC1, PC2): {np.cov(Z_2d.T)[0,1]:.6f}  (should be ≈ 0, decorrelated)")

    section("2. RECONSTRUCTION ERROR")
    for k in [1, 2]:
        pca_k = PCA(n_components=k).fit(X_2d)
        err = pca_k.reconstruction_error(X_2d)
        Z_k = pca_k.transform(X_2d)
        X_rec = pca_k.inverse_transform(Z_k)
        print(f"k={k}: recon error={err:.6f}, variance retained={pca_k.explained_variance_ratio_.sum():.4f}")
    print(f"Note: k=2 retains 100% — perfect reconstruction (2D data)")

    section("3. PCA vs SVD EQUIVALENCE")
    X_test, _ = make_classification(n_samples=200, n_features=8, random_state=0)
    X_test = StandardScaler().fit_transform(X_test)

    pca_eig = PCA(n_components=4).fit(X_test)
    scores_svd, comps_svd, evals_svd = pca_svd(X_test, k=4)
    scores_eig = pca_eig.transform(X_test)

    # Signs may flip — compare absolute values
    match = np.allclose(np.abs(scores_eig), np.abs(scores_svd), atol=1e-6)
    eval_match = np.allclose(np.abs(pca_eig.explained_variance_), np.abs(evals_svd), atol=1e-6)
    print(f"Scores match (eigendecomp vs SVD): {match}")
    print(f"Eigenvalues match:                 {eval_match}")
    print(f"Eigendecomp evals: {pca_eig.explained_variance_.round(4)}")
    print(f"SVD-based evals:   {evals_svd.round(4)}")

    section("4. DIGITS DATASET — HIGH-DIMENSIONAL PCA")
    digits = load_digits()
    X_dig = digits.data.astype(float)    # (1797, 64)
    y_dig = digits.target
    X_dig_s = StandardScaler().fit_transform(X_dig)

    pca_full = PCA().fit(X_dig_s)

    print(f"Original: {X_dig.shape}  (64 features per 8×8 image)")
    print(f"\nComponents needed for variance threshold:")
    for thresh in [0.80, 0.90, 0.95, 0.99]:
        k = pca_full.n_components_for_variance(thresh)
        print(f"  {thresh*100:.0f}% variance → k={k} components  "
              f"(compression {k/64*100:.1f}%)")

    print(f"\nExplained variance by first 10 PCs:")
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    for i in range(10):
        print(f"  PC{i+1:2d}: {pca_full.explained_variance_ratio_[i]*100:.2f}%  "
              f"(cumulative: {cumvar[i]*100:.2f}%)")

    section("5. SKLEARN PCA vs SCRATCH PCA")
    sk_pca = SKLearnPCA(n_components=20).fit(X_dig_s)
    my_pca = PCA(n_components=20).fit(X_dig_s)

    evr_match = np.allclose(
        np.abs(sk_pca.explained_variance_ratio_),
        np.abs(my_pca.explained_variance_ratio_),
        atol=1e-6,
    )
    print(f"EVR match (sklearn vs scratch): {evr_match}")
    print(f"sklearn EVR[:5]: {sk_pca.explained_variance_ratio_[:5].round(4)}")
    print(f"scratch EVR[:5]: {my_pca.explained_variance_ratio_[:5].round(4)}")

    section("6. PCA AS PREPROCESSING — CLASSIFICATION IMPACT")
    split = 1200
    X_tr, X_te = X_dig_s[:split], X_dig_s[split:]
    y_tr, y_te = y_dig[:split], y_dig[split:]

    print(f"{'n_components':>14}  {'Variance':>10}  {'Test Acc':>10}  {'Train time ∝'}")
    print("-" * 55)
    for k in [2, 5, 10, 20, 40, 64]:
        if k < 64:
            pca_k = PCA(n_components=k).fit(X_tr)
            X_tr_k = pca_k.transform(X_tr)
            X_te_k = pca_k.transform(X_te)
            var_ret = pca_k.explained_variance_ratio_.sum()
        else:
            X_tr_k, X_te_k = X_tr, X_te
            var_ret = 1.0
        lr = LogisticRegression(max_iter=500, C=1.0).fit(X_tr_k, y_tr)
        acc = accuracy_score(y_te, lr.predict(X_te_k))
        print(f"{k:>14}  {var_ret:>10.4f}  {acc:>10.4f}  {'d=' + str(k)}")

    section("7. RECONSTRUCTION QUALITY — DIGITS")
    pca_rec = PCA(n_components=20).fit(X_dig_s)
    Z = pca_rec.transform(X_dig_s)
    X_rec = pca_rec.inverse_transform(Z)

    mse_full = np.mean((X_dig_s - X_rec) ** 2)
    psnr = 10 * np.log10(X_dig_s.var() / mse_full)
    print(f"k=20 components on 64-dim digits:")
    print(f"  Reconstruction MSE: {mse_full:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Compression ratio: 64/20 = {64/20:.1f}x")


if __name__ == "__main__":
    main()
