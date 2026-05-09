import numpy as np
from numpy.linalg import inv, norm


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Dataset generation ────────────────────────────────────────

def make_regression(n=200, d=5, noise=0.5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w_true = rng.uniform(-2, 2, d)
    y = X @ w_true + rng.normal(0, noise, n)
    return X, y, w_true


def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


# ── OLS Normal Equation ───────────────────────────────────────

def ols(X, y):
    """w = (X^T X)^{-1} X^T y — closed form."""
    Xb = add_bias(X)
    return inv(Xb.T @ Xb) @ Xb.T @ y


# ── Gradient Descent ─────────────────────────────────────────

def gd_linear_regression(X, y, lr=0.01, n_iters=1000, l2=0.0):
    """
    Mini-batch gradient descent with optional L2 regularization.
    grad = (2/N) X^T(Xw - y) + 2*l2*w  (bias weight not regularized)
    """
    Xb = add_bias(X)
    n, d = Xb.shape
    w = np.zeros(d)
    losses = []

    for _ in range(n_iters):
        residuals = Xb @ w - y
        loss = np.mean(residuals**2) + l2 * np.dot(w[1:], w[1:])
        losses.append(loss)

        grad = (2 / n) * Xb.T @ residuals
        grad[1:] += 2 * l2 * w[1:]   # don't regularize bias
        w -= lr * grad

    return w, losses


# ── Ridge (closed form) ───────────────────────────────────────

def ridge(X, y, lam):
    """w = (X^T X + λI)^{-1} X^T y  (bias column excluded from penalty)."""
    Xb = add_bias(X)
    d = Xb.shape[1]
    reg = lam * np.eye(d)
    reg[0, 0] = 0.0   # don't penalize bias
    return inv(Xb.T @ Xb + reg) @ Xb.T @ y


# ── Lasso via Coordinate Descent ─────────────────────────────

def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)


def lasso_coordinate_descent(X, y, lam, n_iters=500):
    """
    Coordinate descent for Lasso:
    For each j: w_j ← soft_threshold(rho_j, lambda) / z_j
    where rho_j = X_j^T (y - X_{-j} w_{-j}), z_j = ||X_j||^2
    """
    Xb = add_bias(X)
    n, d = Xb.shape
    w = np.zeros(d)

    Xb_sq = np.sum(Xb**2, axis=0)  # precompute column norms squared

    for _ in range(n_iters):
        for j in range(d):
            residual = y - Xb @ w + Xb[:, j] * w[j]
            rho_j = Xb[:, j] @ residual / n
            if j == 0:
                w[j] = rho_j / (Xb_sq[j] / n)   # no penalty on bias
            else:
                w[j] = soft_threshold(rho_j, lam) / (Xb_sq[j] / n)

    return w


# ── Metrics ───────────────────────────────────────────────────

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def predict(w, X):
    return add_bias(X) @ w


# ── Main ──────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    X, y, w_true = make_regression(n=300, d=6, noise=1.0)

    # Train / test split (80/20)
    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    section("1. OLS — NORMAL EQUATION")
    w_ols = ols(X_tr, y_tr)
    y_pred_ols = predict(w_ols, X_te)
    print(f"True weights:  {w_true.round(3)}")
    print(f"OLS weights:   {w_ols[1:].round(3)}  (bias={w_ols[0]:.3f})")
    print(f"Test MSE:  {mse(y_te, y_pred_ols):.4f}")
    print(f"Test R²:   {r2_score(y_te, y_pred_ols):.4f}")

    section("2. GRADIENT DESCENT")
    w_gd, losses_gd = gd_linear_regression(X_tr, y_tr, lr=0.05, n_iters=500)
    y_pred_gd = predict(w_gd, X_te)
    print(f"GD weights:    {w_gd[1:].round(3)}")
    print(f"Test MSE:  {mse(y_te, y_pred_gd):.4f}")
    print(f"Test R²:   {r2_score(y_te, y_pred_gd):.4f}")
    print(f"Loss @ iter 0:   {losses_gd[0]:.4f}")
    print(f"Loss @ iter 499: {losses_gd[-1]:.4f}")
    print(f"OLS vs GD weights match: {np.allclose(w_ols, w_gd, atol=1e-2)}")

    section("3. RIDGE REGRESSION (L2)")
    lambdas = [0.001, 0.1, 1.0, 10.0, 100.0]
    print(f"{'λ':>8}  {'Test MSE':>10}  {'||w||':>8}  {'R²':>8}")
    print("-" * 40)
    for lam in lambdas:
        w_r = ridge(X_tr, y_tr, lam)
        y_pred_r = predict(w_r, X_te)
        print(f"{lam:>8.3f}  {mse(y_te, y_pred_r):>10.4f}  "
              f"{norm(w_r[1:]):>8.4f}  {r2_score(y_te, y_pred_r):>8.4f}")

    section("4. LASSO REGRESSION (L1) — COORDINATE DESCENT")
    lambdas_l1 = [0.001, 0.05, 0.2, 0.5, 1.0]
    print(f"{'λ':>8}  {'Test MSE':>10}  {'||w||_1':>8}  {'Nonzero':>8}")
    print("-" * 45)
    for lam in lambdas_l1:
        w_l = lasso_coordinate_descent(X_tr, y_tr, lam, n_iters=300)
        y_pred_l = predict(w_l, X_te)
        nz = np.sum(np.abs(w_l[1:]) > 1e-4)
        print(f"{lam:>8.3f}  {mse(y_te, y_pred_l):>10.4f}  "
              f"{norm(w_l[1:], 1):>8.4f}  {nz:>8}")

    section("5. REGULARIZATION EFFECT ON WEIGHTS")
    print("True weights:", w_true.round(3))
    print("\nWeight magnitudes across methods:")
    w_methods = {
        "OLS":        ols(X_tr, y_tr)[1:],
        "Ridge(λ=1)": ridge(X_tr, y_tr, 1.0)[1:],
        "Lasso(λ=0.1)": lasso_coordinate_descent(X_tr, y_tr, 0.1)[1:],
    }
    for name, w in w_methods.items():
        print(f"  {name:15s}: {w.round(3)}")

    section("6. MULTICOLLINEARITY DEMO")
    # Create correlated features: x2 ≈ x1
    X_mc = X_tr.copy()
    X_mc[:, 1] = X_mc[:, 0] + np.random.normal(0, 0.01, len(X_mc))

    w_ols_mc = ols(X_mc, y_tr)
    w_ridge_mc = ridge(X_mc, y_tr, lam=1.0)
    cond_plain = np.linalg.cond(add_bias(X_mc).T @ add_bias(X_mc))
    cond_ridge = np.linalg.cond(add_bias(X_mc).T @ add_bias(X_mc) + 1.0 * np.eye(X_mc.shape[1] + 1))

    print(f"Condition number (X^TX):         {cond_plain:.2e}  (high → ill-conditioned)")
    print(f"Condition number (X^TX + λI):    {cond_ridge:.2e}  (Ridge stabilizes)")
    print(f"OLS weights (x0, x1):   {w_ols_mc[1:3].round(2)}  (can blow up)")
    print(f"Ridge weights (x0, x1): {w_ridge_mc[1:3].round(2)}  (stable)")


if __name__ == "__main__":
    main()
