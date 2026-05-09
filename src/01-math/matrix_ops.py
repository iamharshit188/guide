import numpy as np
from numpy.linalg import eig, svd, inv, matrix_rank, norm


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    section("1. MATRIX MULTIPLICATION")

    A = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=float)  # (2, 3)
    B = np.array([[7,  8],
                  [9,  10],
                  [11, 12]], dtype=float)  # (3, 2)

    C = A @ B  # (2, 2)
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"C = A @ B:\n{C}")
    print(f"C[0,0] = 1*7 + 2*9 + 3*11 = {1*7 + 2*9 + 3*11}  (manual check)")

    # Element-wise vs matrix multiply
    D = np.array([[1, 2], [3, 4]], dtype=float)
    E = np.array([[5, 6], [7, 8]], dtype=float)
    print(f"\nElement-wise D*E:\n{D * E}")
    print(f"Matrix D@E:\n{D @ E}")

    section("2. TRANSPOSE & SYMMETRY")

    print(f"A:\n{A}")
    print(f"A.T:\n{A.T}")
    print(f"(A@B).T == B.T @ A.T: {np.allclose((A@B).T, B.T @ A.T)}")

    S = D @ D.T  # symmetric by construction
    print(f"\nS = D @ D.T (symmetric):\n{S}")
    print(f"S == S.T: {np.allclose(S, S.T)}")

    section("3. MATRIX INVERSE")

    M = np.array([[2, 1], [5, 3]], dtype=float)
    M_inv = inv(M)

    print(f"M:\n{M}")
    print(f"M_inv:\n{M_inv}")
    print(f"M @ M_inv:\n{(M @ M_inv).round(10)}  (should be I)")

    # Condition number: ratio largest/smallest singular value
    cond = np.linalg.cond(M)
    print(f"Condition number: {cond:.4f}  (higher → more ill-conditioned)")

    section("4. DETERMINANT & RANK")

    print(f"det(M) = {np.linalg.det(M):.4f}")
    print(f"rank(A) = {matrix_rank(A)}")

    singular = np.array([[1, 2], [2, 4]], dtype=float)  # row2 = 2*row1
    print(f"\nSingular matrix:\n{singular}")
    print(f"det = {np.linalg.det(singular):.4f}  (→ 0, not invertible)")
    print(f"rank = {matrix_rank(singular)}")

    section("5. EIGENDECOMPOSITION")

    # Symmetric matrix → real eigenvalues, orthogonal eigenvectors
    cov = np.array([[4, 2],
                    [2, 3]], dtype=float)

    eigenvalues, eigenvectors = eig(cov)

    print(f"Covariance matrix:\n{cov}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors (columns):\n{eigenvectors}")

    # Verify: A*v = λ*v
    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        v = eigenvectors[:, i]
        Av = cov @ v
        lv = lam * v
        print(f"\nλ={lam:.4f}: A*v = {Av.round(4)}, λ*v = {lv.round(4)}, match={np.allclose(Av, lv)}")

    # Reconstruction: A = V @ diag(λ) @ V^-1
    V = eigenvectors
    L = np.diag(eigenvalues)
    A_reconstructed = V @ L @ inv(V)
    print(f"\nReconstruction A = V Λ V⁻¹ matches original: {np.allclose(A_reconstructed, cov)}")

    # Trace = sum of eigenvalues
    print(f"Trace(A) = {np.trace(cov):.4f}, sum(eigenvalues) = {sum(eigenvalues):.4f}")
    # Det = product of eigenvalues
    print(f"det(A) = {np.linalg.det(cov):.4f}, prod(eigenvalues) = {np.prod(eigenvalues):.4f}")

    section("6. SVD — SINGULAR VALUE DECOMPOSITION")

    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]], dtype=float)  # (4, 3)

    U, sigma, Vt = svd(X, full_matrices=True)
    Sigma = np.zeros_like(X)
    np.fill_diagonal(Sigma, sigma)

    print(f"X shape: {X.shape}")
    print(f"U: {U.shape}, Σ: {Sigma.shape}, Vt: {Vt.shape}")
    print(f"Singular values: {sigma.round(4)}")
    print(f"X reconstructed from U Σ Vt: {np.allclose(U @ Sigma @ Vt, X)}")

    # Orthonormality of U and V
    print(f"U.T @ U ≈ I: {np.allclose(U.T @ U, np.eye(U.shape[0]))}")
    print(f"Vt @ Vt.T ≈ I: {np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]))}")

    # Rank-k approximation
    for k in [1, 2]:
        U_k = U[:, :k]
        S_k = np.diag(sigma[:k])
        Vt_k = Vt[:k, :]
        X_k = U_k @ S_k @ Vt_k

        fro_err = norm(X - X_k, 'fro')
        fro_orig = norm(X, 'fro')
        variance_explained = (sigma[:k]**2).sum() / (sigma**2).sum() * 100

        print(f"\nRank-{k} approximation:")
        print(f"  Frobenius error: {fro_err:.4f} / {fro_orig:.4f}")
        print(f"  Variance explained: {variance_explained:.1f}%")

    section("7. MOORE-PENROSE PSEUDOINVERSE")

    # Overdetermined system: more equations than unknowns → least-squares solution
    # Ax ≈ b, A is (m x n) with m > n
    A_over = np.array([[1, 1],
                       [1, 2],
                       [1, 3],
                       [1, 4]], dtype=float)
    b = np.array([1.5, 2.5, 3.8, 5.1])

    # Pseudoinverse: A+ = (A.T A)^-1 A.T (for full column rank)
    A_pinv = np.linalg.pinv(A_over)
    x_lstsq = A_pinv @ b

    print(f"A (overdetermined) shape: {A_over.shape}")
    print(f"Least-squares solution x = A⁺ b: {x_lstsq.round(4)}")
    print(f"Residual ||Ax - b||_2 = {norm(A_over @ x_lstsq - b):.4f}")

    # Verify against np.linalg.lstsq
    x_ref, _, _, _ = np.linalg.lstsq(A_over, b, rcond=None)
    print(f"np.lstsq solution: {x_ref.round(4)}")
    print(f"Solutions match: {np.allclose(x_lstsq, x_ref)}")

    section("8. PCA VIA EIGENDECOMPOSITION")

    np.random.seed(42)
    n_samples = 200
    # Correlated 2D data
    cov_true = np.array([[3, 1.5], [1.5, 1]])
    data = np.random.multivariate_normal([0, 0], cov_true, n_samples)

    # Center
    data_centered = data - data.mean(axis=0)

    # Covariance matrix
    C = (data_centered.T @ data_centered) / (n_samples - 1)

    # Eigendecomp of covariance
    evals, evecs = eig(C)
    idx = np.argsort(evals)[::-1]  # sort descending
    evals, evecs = evals[idx], evecs[:, idx]

    print(f"Estimated covariance:\n{C.round(3)}")
    print(f"True covariance:\n{cov_true}")
    print(f"PC1 (largest variance direction): {evecs[:, 0].round(4)}")
    print(f"Variance explained: PC1={evals[0]/evals.sum()*100:.1f}%, PC2={evals[1]/evals.sum()*100:.1f}%")

    # Project onto PC1 and PC2
    projected = data_centered @ evecs
    print(f"Projected data shape: {projected.shape}")
    print(f"Variance in PC1 direction: {projected[:, 0].var():.4f}  (should ≈ {evals[0]:.4f})")


if __name__ == "__main__":
    main()
