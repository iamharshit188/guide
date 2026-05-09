import numpy as np
from scipy import stats


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── 1. Basic Probability ──────────────────────────────────────

def basic_probability():
    section("1. BASIC PROBABILITY & BAYES")

    # Spam classifier example: P(spam|word) via Bayes
    # P(spam) = 0.3, P(word="free"|spam) = 0.8, P(word="free"|ham) = 0.1
    p_spam = 0.3
    p_ham = 1 - p_spam
    p_word_given_spam = 0.8
    p_word_given_ham = 0.1

    # P(word) via total probability
    p_word = p_word_given_spam * p_spam + p_word_given_ham * p_ham
    # Bayes: P(spam|word) = P(word|spam)*P(spam) / P(word)
    p_spam_given_word = (p_word_given_spam * p_spam) / p_word

    print("Spam classifier (Naive Bayes):")
    print(f"  P(spam) = {p_spam}")
    print(f"  P('free'|spam) = {p_word_given_spam}")
    print(f"  P('free'|ham)  = {p_word_given_ham}")
    print(f"  P('free')      = {p_word:.4f}  (total probability)")
    print(f"  P(spam|'free') = {p_spam_given_word:.4f}  ← posterior")

    # Sequential Bayes update (coin flip)
    print("\nSequential Bayes update (fair coin test):")
    print("Prior: P(fair) = 0.5, P(biased) = 0.5")
    print("Biased coin: P(H|biased) = 0.9, fair: P(H|fair) = 0.5")

    p_fair = 0.5
    p_biased = 0.5
    p_h_fair = 0.5
    p_h_biased = 0.9

    for flip, outcome in enumerate(["H", "H", "H", "T", "H"]):
        if outcome == "H":
            lik_fair, lik_biased = p_h_fair, p_h_biased
        else:
            lik_fair, lik_biased = 1 - p_h_fair, 1 - p_h_biased

        p_data = lik_fair * p_fair + lik_biased * p_biased
        p_fair = lik_fair * p_fair / p_data
        p_biased = lik_biased * p_biased / p_data
        print(f"  After '{outcome}': P(fair)={p_fair:.4f}, P(biased)={p_biased:.4f}")


# ── 2. Key Distributions ─────────────────────────────────────

def distributions():
    section("2. KEY DISTRIBUTIONS")

    np.random.seed(42)

    # Bernoulli
    print("Bernoulli(p=0.3):")
    samples = np.random.binomial(1, 0.3, size=10000)
    print(f"  E[X] = p = 0.3,  sample mean = {samples.mean():.4f}")
    print(f"  Var(X) = p(1-p) = 0.21,  sample var = {samples.var():.4f}")

    # Categorical (Multinomial)
    print("\nCategorical(p=[0.2, 0.5, 0.3]):")
    probs = np.array([0.2, 0.5, 0.3])
    samples_cat = np.random.choice([0, 1, 2], size=10000, p=probs)
    for k, (true_p, emp_p) in enumerate(zip(probs, np.bincount(samples_cat) / 10000)):
        print(f"  class {k}: true={true_p:.2f}, empirical={emp_p:.4f}")

    # Gaussian
    print("\nGaussian(μ=5, σ²=4):")
    mu, sigma = 5.0, 2.0
    samples_g = np.random.normal(mu, sigma, 100000)
    print(f"  True:    μ={mu}, σ²={sigma**2}")
    print(f"  Empirical: μ={samples_g.mean():.4f}, σ²={samples_g.var():.4f}")
    print(f"  68% of data in [μ-σ, μ+σ]: {((samples_g >= mu-sigma) & (samples_g <= mu+sigma)).mean():.4f}  (expect 0.683)")
    print(f"  95% of data in [μ-2σ, μ+2σ]: {((samples_g >= mu-2*sigma) & (samples_g <= mu+2*sigma)).mean():.4f}  (expect 0.954)")

    # PDF evaluation
    x = np.array([3.0, 5.0, 7.0])
    pdf_vals = stats.norm.pdf(x, loc=mu, scale=sigma)
    print(f"  PDF at x={x}: {pdf_vals.round(4)}")

    # Standard Normal — Z-score
    print("\nZ-score standardization:")
    raw = np.array([10, 20, 30, 40, 50], dtype=float)
    z = (raw - raw.mean()) / raw.std()
    print(f"  raw:     {raw}")
    print(f"  z-scores: {z.round(4)}")
    print(f"  z mean={z.mean():.4f}, z std={z.std():.4f}")

    # Poisson
    print("\nPoisson(λ=3):")
    lam = 3
    samples_p = np.random.poisson(lam, size=10000)
    print(f"  True: E[X]=Var[X]=λ={lam}")
    print(f"  Empirical: mean={samples_p.mean():.4f}, var={samples_p.var():.4f}")


# ── 3. Maximum Likelihood Estimation ─────────────────────────

def mle_demo():
    section("3. MAXIMUM LIKELIHOOD ESTIMATION")

    np.random.seed(0)
    true_mu, true_sigma = 7.0, 2.5
    n_samples = 1000
    data = np.random.normal(true_mu, true_sigma, n_samples)

    # MLE for Gaussian
    # ℓ(μ,σ²) = -N/2 * log(2πσ²) - 1/(2σ²) * Σ(xᵢ-μ)²
    # ∂ℓ/∂μ = 0  → μ̂ = (1/N)Σxᵢ
    # ∂ℓ/∂σ² = 0 → σ̂² = (1/N)Σ(xᵢ-μ̂)²
    mu_mle = data.mean()
    sigma2_mle = data.var()         # biased MLE
    sigma2_unbiased = data.var(ddof=1)  # unbiased (Bessel's correction)

    def log_likelihood(mu, sigma2, x):
        n = len(x)
        return (-n/2) * np.log(2 * np.pi * sigma2) - (1 / (2*sigma2)) * np.sum((x - mu)**2)

    ll_true = log_likelihood(true_mu, true_sigma**2, data)
    ll_mle = log_likelihood(mu_mle, sigma2_mle, data)

    print(f"Data: N={n_samples}, true μ={true_mu}, true σ²={true_sigma**2}")
    print(f"\nMLE estimates:")
    print(f"  μ̂_MLE = {mu_mle:.4f}  (true: {true_mu})")
    print(f"  σ̂²_MLE = {sigma2_mle:.4f}  (true: {true_sigma**2})  [biased]")
    print(f"  σ̂²_unbiased = {sigma2_unbiased:.4f}  (Bessel's correction N-1)")

    print(f"\nLog-likelihood:")
    print(f"  ℓ(true params)  = {ll_true:.2f}")
    print(f"  ℓ(MLE params)   = {ll_mle:.2f}  (should be ≥ true)")

    # Bernoulli MLE: p̂ = (# successes) / N
    print("\nBernoulli MLE:")
    true_p = 0.65
    coin_flips = np.random.binomial(1, true_p, 500)
    p_mle = coin_flips.mean()
    print(f"  True p={true_p}, MLE p̂={p_mle:.4f}")

    # Grid search over log-likelihood to show it's maximized at MLE
    p_grid = np.linspace(0.01, 0.99, 200)
    ll_grid = [np.sum(np.log(p**coin_flips * (1-p)**(1-coin_flips))) for p in p_grid]
    p_grid_max = p_grid[np.argmax(ll_grid)]
    print(f"  Grid search argmax ℓ(p) = {p_grid_max:.4f}  (should ≈ p̂_MLE)")


# ── 4. Entropy, Cross-Entropy, KL Divergence ─────────────────

def information_theory():
    section("4. INFORMATION THEORY")

    def entropy(p):
        """H(P) = -Σ p(x) log p(x)"""
        p = np.array(p, dtype=float)
        p = p[p > 0]  # 0 log 0 = 0
        return -np.sum(p * np.log2(p))

    def cross_entropy(p, q):
        """H(P,Q) = -Σ p(x) log q(x)"""
        p, q = np.array(p, dtype=float), np.array(q, dtype=float)
        return -np.sum(p * np.log2(q + 1e-12))

    def kl_divergence(p, q):
        """D_KL(P||Q) = Σ p(x) log(p(x)/q(x)) = H(P,Q) - H(P)"""
        p, q = np.array(p, dtype=float), np.array(q, dtype=float)
        return np.sum(p * np.log2((p + 1e-12) / (q + 1e-12)))

    print("Entropy examples:")
    print(f"  Uniform 4-class: H = {entropy([0.25,0.25,0.25,0.25]):.4f} bits  (max = log2(4) = 2.0)")
    print(f"  Skewed [0.9,0.1]: H = {entropy([0.9, 0.1]):.4f} bits")
    print(f"  Deterministic [1,0]: H = {entropy([1, 0]):.4f} bits  (no uncertainty)")

    print("\nCross-entropy as classification loss:")
    # True labels (one-hot) and model predictions
    true_dist = np.array([0.0, 1.0, 0.0])  # class 1 is correct
    perfect_pred = np.array([0.01, 0.98, 0.01])
    bad_pred = np.array([0.33, 0.34, 0.33])
    wrong_pred = np.array([0.01, 0.01, 0.98])

    for name, pred in [("perfect", perfect_pred), ("random", bad_pred), ("wrong", wrong_pred)]:
        ce = cross_entropy(true_dist, pred)
        print(f"  CE({name}): {ce:.4f} bits  [pred={pred}]")

    print("\nKL Divergence:")
    p = np.array([0.4, 0.4, 0.2])
    q1 = np.array([0.4, 0.4, 0.2])  # identical
    q2 = np.array([0.25, 0.25, 0.5])  # different
    q3 = np.array([0.1, 0.8, 0.1])  # another

    print(f"  D_KL(P||P) = {kl_divergence(p, q1):.4f}  (should be 0)")
    print(f"  D_KL(P||Q2) = {kl_divergence(p, q2):.4f}")
    print(f"  D_KL(Q2||P) = {kl_divergence(q2, p):.4f}  (asymmetric!)")
    print(f"  D_KL(P||Q3) = {kl_divergence(p, q3):.4f}")

    print("\nRelationship: H(P,Q) = H(P) + D_KL(P||Q)")
    h_p = entropy(p)
    h_pq = cross_entropy(p, q2)
    kl_pq = kl_divergence(p, q2)
    print(f"  H(P) = {h_p:.4f}")
    print(f"  D_KL(P||Q2) = {kl_pq:.4f}")
    print(f"  H(P) + D_KL = {h_p + kl_pq:.4f}")
    print(f"  H(P, Q2) = {h_pq:.4f}  ✓" if abs(h_p + kl_pq - h_pq) < 1e-6 else "  Mismatch!")


# ── 5. Covariance & Multivariate Gaussian ────────────────────

def covariance_demo():
    section("5. COVARIANCE & MULTIVARIATE GAUSSIAN")

    np.random.seed(7)
    n = 1000

    # Correlated data
    mean = [2.0, 5.0]
    cov_true = [[3.0, 2.0],
                [2.0, 2.0]]

    data = np.random.multivariate_normal(mean, cov_true, n)
    x, y = data[:, 0], data[:, 1]

    cov_empirical = np.cov(data.T)  # (2, 2)
    corr = np.corrcoef(data.T)

    print(f"True covariance:\n{np.array(cov_true)}")
    print(f"Empirical covariance:\n{cov_empirical.round(3)}")
    print(f"Pearson correlation:\n{corr.round(3)}")
    print(f"  ρ(x,y) = Cov(x,y) / (σ_x * σ_y) = {cov_empirical[0,1]:.3f} / ({x.std():.3f} * {y.std():.3f}) = {corr[0,1]:.4f}")

    # Mahalanobis distance vs Euclidean
    cov_mat = np.cov(data.T)
    cov_inv = np.linalg.inv(cov_mat)
    mu_hat = data.mean(axis=0)

    test_point = np.array([4.0, 7.0])
    diff = test_point - mu_hat

    euclidean = np.linalg.norm(diff)
    mahalanobis = np.sqrt(diff @ cov_inv @ diff)

    print(f"\nTest point: {test_point}, Data mean: {mu_hat.round(3)}")
    print(f"  Euclidean distance:   {euclidean:.4f}")
    print(f"  Mahalanobis distance: {mahalanobis:.4f}  (accounts for correlation/scale)")

    # PDF evaluation at mean vs test point
    rv = stats.multivariate_normal(mean=mu_hat, cov=cov_mat)
    print(f"\nMultivariate Gaussian PDF:")
    print(f"  p(mean)       = {rv.pdf(mu_hat):.6f}  (should be max)")
    print(f"  p(test_point) = {rv.pdf(test_point):.6f}")


def main():
    basic_probability()
    distributions()
    mle_demo()
    information_theory()
    covariance_demo()

    section("SUMMARY")
    print("Key identities to internalize:")
    print("  • P(A|B) = P(B|A)*P(A) / P(B)                 [Bayes]")
    print("  • E[X] = Σ x*p(x)                              [expectation]")
    print("  • Var(X) = E[X²] - (E[X])²                    [variance shortcut]")
    print("  • MLE: maximize Σ log p(xᵢ; θ)                [log-likelihood]")
    print("  • H(P) = -Σ p log p                            [entropy]")
    print("  • CE = H(P,Q) = H(P) + D_KL(P||Q)             [decomposition]")
    print("  • Minimizing CE loss ≡ minimizing KL(true||model)")


if __name__ == "__main__":
    main()
