import numpy as np
from sklearn.datasets import make_classification, make_regression, load_breast_cancer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from sklearn.preprocessing import StandardScaler


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Gradient Boosting from scratch (regression) ───────────────

class GradientBoostingRegressorScratch:
    """
    Gradient Boosting for MSE loss.
    F_m(x) = F_{m-1}(x) + η * h_m(x)
    h_m fits negative gradient = residuals = y - F_{m-1}(x)
    """

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, subsample=1.0, seed=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.seed = seed
        self.trees_ = []
        self.F0_ = None
        self.train_losses_ = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(y)

        # Initial prediction: mean (optimal for MSE)
        self.F0_ = y.mean()
        F = np.full(n, self.F0_)

        for m in range(self.n_estimators):
            # Negative gradient for MSE: r = y - F
            residuals = y - F
            self.train_losses_.append(np.mean(residuals**2))

            # Subsample rows
            if self.subsample < 1.0:
                idx = rng.choice(n, int(n * self.subsample), replace=False)
                X_sub, r_sub = X[idx], residuals[idx]
            else:
                X_sub, r_sub = X, residuals

            # Fit weak learner to pseudo-residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=m)
            tree.fit(X_sub, r_sub)
            self.trees_.append(tree)

            # Update ensemble
            F += self.learning_rate * tree.predict(X)

        self.train_losses_.append(np.mean((y - F)**2))
        return self

    def predict(self, X):
        F = np.full(len(X), self.F0_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    def staged_predict(self, X):
        """Yield predictions at each boosting stage."""
        F = np.full(len(X), self.F0_)
        yield F.copy()
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
            yield F.copy()


class GradientBoostingClassifierScratch:
    """
    Gradient Boosting for binary log-loss.
    Pseudo-residuals: r_i = y_i - σ(F(x_i))
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, seed=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.seed = seed
        self.trees_ = []
        self.F0_ = None

    @staticmethod
    def _sigmoid(z):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def fit(self, X, y):
        n = len(y)
        # Initial log-odds: log(p/(1-p)) where p = mean(y)
        p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.F0_ = np.log(p0 / (1 - p0))
        F = np.full(n, self.F0_)

        for m in range(self.n_estimators):
            p = self._sigmoid(F)
            residuals = y - p   # negative gradient of log-loss

            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=m)
            tree.fit(X, residuals)
            self.trees_.append(tree)
            F += self.learning_rate * tree.predict(X)

        return self

    def predict_proba(self, X):
        F = np.full(len(X), self.F0_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        p = self._sigmoid(F)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ── Feature importance from trees ─────────────────────────────

def feature_importance_from_gb(gb_model, X_tr):
    """Average impurity decrease across all trees and all nodes."""
    importances = np.zeros(X_tr.shape[1])
    for tree in gb_model.trees_:
        fi = tree.feature_importances_
        importances += fi
    return importances / len(gb_model.trees_)


# ── Main ──────────────────────────────────────────────────────

def main():
    section("1. GRADIENT BOOSTING REGRESSION FROM SCRATCH")
    X_reg, y_reg = make_regression(n_samples=500, n_features=10, n_informative=7,
                                   noise=15, random_state=42)
    X_rtr, X_rte, y_rtr, y_rte = train_test_split(X_reg, y_reg, test_size=0.2, random_state=0)

    gb_reg = GradientBoostingRegressorScratch(
        n_estimators=200, learning_rate=0.05, max_depth=3, seed=42
    ).fit(X_rtr, y_rtr)

    y_pred_reg = gb_reg.predict(X_rte)
    mse_test = mean_squared_error(y_rte, y_pred_reg)
    rmse_test = np.sqrt(mse_test)

    print(f"Regression (n=500, d=10, noise=15):")
    print(f"  Test RMSE: {rmse_test:.4f}")
    print(f"  Baseline (predict mean) RMSE: {np.std(y_rte):.4f}")
    print(f"\nTraining loss progression:")
    for i in [0, 10, 25, 50, 100, 199]:
        print(f"  Iter {i:3d}: train_MSE = {gb_reg.train_losses_[i]:.4f}")

    section("2. STAGED PREDICTION — OPTIMAL N_ESTIMATORS")
    sk_gb_reg = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=42
    ).fit(X_rtr, y_rtr)

    print(f"{'n_trees':>8}  {'Train RMSE':>12}  {'Test RMSE':>12}")
    print("-" * 40)
    stages = [10, 25, 50, 100, 150, 200, 300]
    for n_trees, pred_staged in zip(
        stages,
        [p for i, p in enumerate(sk_gb_reg.staged_predict(X_rte)) if i+1 in stages]
    ):
        pred_tr_staged = list(sk_gb_reg.staged_predict(X_rtr))[n_trees - 1]
        rmse_tr = np.sqrt(mean_squared_error(y_rtr, pred_tr_staged))
        rmse_te = np.sqrt(mean_squared_error(y_rte, pred_staged))
        print(f"{n_trees:>8}  {rmse_tr:>12.4f}  {rmse_te:>12.4f}")

    section("3. GRADIENT BOOSTING CLASSIFICATION FROM SCRATCH")
    X_cls, y_cls = make_classification(n_samples=600, n_features=8, n_informative=5,
                                       random_state=1)
    X_ctr, X_cte, y_ctr, y_cte = train_test_split(X_cls, y_cls, test_size=0.25, random_state=0)

    gb_cls = GradientBoostingClassifierScratch(
        n_estimators=150, learning_rate=0.1, max_depth=3, seed=42
    ).fit(X_ctr, y_ctr)

    acc = accuracy_score(y_cte, gb_cls.predict(X_cte))
    proba = gb_cls.predict_proba(X_cte)[:, 1]
    ll = log_loss(y_cte, proba)
    print(f"Binary classification (scratch GBC):")
    print(f"  Test accuracy:  {acc:.4f}")
    print(f"  Test log-loss:  {ll:.4f}")

    section("4. SKLEARN GBC — HYPERPARAMETER IMPACT")
    print(f"{'lr':>6}  {'n_est':>6}  {'depth':>6}  {'subsamp':>8}  {'Test Acc':>10}  {'Log-loss':>10}")
    print("-" * 58)
    configs = [
        (0.3, 50, 3, 1.0),
        (0.1, 100, 3, 1.0),
        (0.1, 100, 3, 0.8),
        (0.05, 200, 3, 0.8),
        (0.05, 200, 5, 0.8),
        (0.01, 500, 3, 0.8),
    ]
    for lr, n_est, depth, sub in configs:
        gb = GradientBoostingClassifier(
            n_estimators=n_est, learning_rate=lr, max_depth=depth,
            subsample=sub, random_state=0
        ).fit(X_ctr, y_ctr)
        acc_g = accuracy_score(y_cte, gb.predict(X_cte))
        ll_g = log_loss(y_cte, gb.predict_proba(X_cte))
        print(f"{lr:>6.2f}  {n_est:>6}  {depth:>6}  {sub:>8.1f}  {acc_g:>10.4f}  {ll_g:>10.4f}")

    section("5. EARLY STOPPING VIA STAGED EVALUATION")
    gb_early = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=0, validation_fraction=0.2,
        n_iter_no_change=15, tol=1e-4
    ).fit(X_ctr, y_ctr)

    print(f"Fitted with 500 max estimators, early stopping (patience=15):")
    print(f"  Actual n_estimators used: {gb_early.n_estimators_}")
    print(f"  Test accuracy: {accuracy_score(y_cte, gb_early.predict(X_cte)):.4f}")

    section("6. FEATURE IMPORTANCE (GBC)")
    bc = load_breast_cancer()
    X_bc, y_bc = bc.data, bc.target
    X_bc_tr, X_bc_te, y_bc_tr, y_bc_te = train_test_split(X_bc, y_bc, test_size=0.2, random_state=0)

    gb_bc = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=42
    ).fit(X_bc_tr, y_bc_tr)

    print(f"Breast cancer GBC:")
    print(f"  Test accuracy: {accuracy_score(y_bc_te, gb_bc.predict(X_bc_te)):.4f}")
    print(f"\nTop 8 features by MDI importance:")
    print(f"  {'Feature':35s}  {'Importance':>12}")
    print("  " + "-" * 52)
    top_idx = np.argsort(gb_bc.feature_importances_)[::-1][:8]
    for i in top_idx:
        print(f"  {bc.feature_names[i]:35s}  {gb_bc.feature_importances_[i]:>12.4f}")

    section("7. GRADIENT BOOSTING vs RANDOM FOREST COMPARISON")
    print("Algorithm comparison on breast cancer dataset:")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, lr=0.05, max_depth=3, random_state=0),
    }

    # Use StandardScaler for LogReg only
    from sklearn.pipeline import Pipeline
    scaler = StandardScaler()

    print(f"{'Model':22s}  {'CV Acc (5-fold)':>16}  {'Test Acc':>10}")
    print("-" * 55)
    for name, mdl in models.items():
        if "Logistic" in name:
            pipe = Pipeline([("sc", StandardScaler()), ("m", mdl)])
            cv_scores = cross_val_score(pipe, X_bc_tr, y_bc_tr, cv=5, scoring="accuracy")
            pipe.fit(X_bc_tr, y_bc_tr)
            te_acc = accuracy_score(y_bc_te, pipe.predict(X_bc_te))
        else:
            cv_scores = cross_val_score(mdl, X_bc_tr, y_bc_tr, cv=5, scoring="accuracy")
            mdl.fit(X_bc_tr, y_bc_tr)
            te_acc = accuracy_score(y_bc_te, mdl.predict(X_bc_te))
        print(f"{name:22s}  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  {te_acc:>10.4f}")

    section("8. XGBOOST MATH — SECOND ORDER APPROXIMATION")
    print("XGBoost splits by maximizing Gain:")
    print("  Gain = 0.5 * [(G_L²/(H_L+λ)) + (G_R²/(H_R+λ)) - (G²/(H+λ))] - γ")
    print("  G_L = Σ_{i∈L} g_i  (sum of gradients in left leaf)")
    print("  H_L = Σ_{i∈L} h_i  (sum of hessians in left leaf)")
    print("  Optimal leaf weight: w* = -G / (H + λ)")
    print("  Pruning: only split if Gain > 0 (controlled by γ)")
    print("")
    print("For log-loss (binary classification):")
    print("  g_i = σ(F(x_i)) - y_i  (1st derivative)")
    print("  h_i = σ(F(x_i)) * (1 - σ(F(x_i)))  (2nd derivative)")
    print("")

    # Demonstrate with sklearn's GBC which also uses 2nd order
    # (True XGBoost requires xgboost package)
    gb_2nd = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0
    ).fit(X_bc_tr, y_bc_tr)
    print(f"sklearn GBC test accuracy (uses 2nd-order approximation): "
          f"{accuracy_score(y_bc_te, gb_bc.predict(X_bc_te)):.4f}")
    print("For true XGBoost: pip install xgboost && import xgboost as xgb")


if __name__ == "__main__":
    main()
