import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import DBSCAN, KMeans as SKLearnKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── K-Means from scratch ──────────────────────────────────────

class KMeans:
    """Lloyd's algorithm with K-Means++ initialization."""

    def __init__(self, k, n_init=10, max_iter=300, tol=1e-4, seed=42):
        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids_pp(self, X, rng):
        """K-Means++ seeding: P(choosing x) ∝ d²(x, nearest centroid)."""
        n = len(X)
        first = rng.integers(0, n)
        centroids = [X[first]]
        for _ in range(self.k - 1):
            dists = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])
            probs = dists / dists.sum()
            centroids.append(X[rng.choice(n, p=probs)])
        return np.array(centroids)

    def _assign(self, X, centroids):
        """Assign each point to nearest centroid."""
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # (N, k)
        return np.argmin(dists, axis=1)

    def _update(self, X, labels):
        """Recompute centroids as cluster means."""
        return np.array([X[labels == k].mean(axis=0) if (labels == k).any()
                         else self.centroids_[k] for k in range(self.k)])

    def _run_once(self, X, rng):
        centroids = self._init_centroids_pp(X, rng)
        for _ in range(self.max_iter):
            labels = self._assign(X, centroids)
            new_centroids = self._update(X, labels)
            if np.linalg.norm(new_centroids - centroids) < self.tol:
                break
            centroids = new_centroids
        inertia = sum(np.sum((X[labels == k] - centroids[k])**2) for k in range(self.k))
        return centroids, labels, inertia

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        best_inertia = float("inf")
        for _ in range(self.n_init):
            centroids, labels, inertia = self._run_once(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                self.centroids_ = centroids
                self.labels_ = labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)


# ── Elbow method ─────────────────────────────────────────────

def elbow_method(X, k_range):
    inertias = []
    for k in k_range:
        km = KMeans(k=k, n_init=5, seed=0).fit(X)
        inertias.append(km.inertia_)
    return inertias


def find_elbow(inertias, k_range):
    """
    Knee/elbow via maximum curvature (second-difference).
    Returns k at max second difference.
    """
    diffs2 = np.diff(inertias, n=2)
    return k_range[np.argmax(diffs2) + 1]


# ── Silhouette score from scratch ─────────────────────────────

def silhouette_score_scratch(X, labels):
    """
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    a(i) = mean intra-cluster dist, b(i) = min mean inter-cluster dist
    """
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    scores = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            scores.append(0.0)
            continue
        a_i = np.mean(np.linalg.norm(X[same] - X[i], axis=1))
        b_i = float("inf")
        for k in unique_labels:
            if k == labels[i]:
                continue
            mask = labels == k
            mean_dist = np.mean(np.linalg.norm(X[mask] - X[i], axis=1))
            b_i = min(b_i, mean_dist)
        scores.append((b_i - a_i) / max(a_i, b_i))
    return float(np.mean(scores))


# ── Main ──────────────────────────────────────────────────────

def main():
    section("1. K-MEANS FROM SCRATCH — WELL-SEPARATED CLUSTERS")
    X_blobs, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=42)
    X_blobs = StandardScaler().fit_transform(X_blobs)

    km = KMeans(k=4, n_init=10, seed=42).fit(X_blobs)
    ari = adjusted_rand_score(y_true, km.labels_)
    sil = silhouette_score(X_blobs, km.labels_)
    print(f"K=4 blobs dataset (true K=4):")
    print(f"  Inertia (WCSS):  {km.inertia_:.4f}")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}  (1.0 = perfect)")
    print(f"  Silhouette score: {sil:.4f}  (−1..1, higher=better)")
    print(f"  Cluster sizes: {[int((km.labels_==k).sum()) for k in range(4)]}")
    print(f"  Centroids:\n{km.centroids_.round(3)}")

    # Compare with sklearn
    sk_km = SKLearnKMeans(n_clusters=4, n_init=10, random_state=42).fit(X_blobs)
    print(f"\n  sklearn KMeans inertia: {sk_km.inertia_:.4f}")
    print(f"  ARI vs sklearn labels: {adjusted_rand_score(km.labels_, sk_km.labels_):.4f}")

    section("2. ELBOW METHOD — OPTIMAL K SELECTION")
    k_range = list(range(1, 9))
    inertias = elbow_method(X_blobs, k_range)
    print(f"{'K':>4}  {'Inertia':>12}  {'Δ Inertia':>12}  {'Δ²':>10}")
    print("-" * 45)
    for i, (k, iner) in enumerate(zip(k_range, inertias)):
        delta = inertias[i-1] - iner if i > 0 else "-"
        delta2 = (inertias[i-2] - inertias[i-1]) - (inertias[i-1] - iner) if i > 1 else "-"
        delta_s = f"{delta:12.2f}" if isinstance(delta, float) else f"{'':>12}"
        delta2_s = f"{delta2:10.2f}" if isinstance(delta2, float) else f"{'':>10}"
        print(f"{k:>4}  {iner:>12.2f}  {delta_s}  {delta2_s}")
    elbow_k = find_elbow(inertias, k_range)
    print(f"\n  Elbow detected at K = {elbow_k}")

    section("3. SILHOUETTE ANALYSIS")
    print(f"{'K':>4}  {'Sil (scratch)':>14}  {'Sil (sklearn)':>14}  {'ARI':>8}")
    print("-" * 50)
    for k in range(2, 7):
        km_k = KMeans(k=k, n_init=5, seed=42).fit(X_blobs)
        sil_s = silhouette_score_scratch(X_blobs, km_k.labels_)
        sil_sk = silhouette_score(X_blobs, km_k.labels_)
        ari_k = adjusted_rand_score(y_true, km_k.labels_)
        print(f"{k:>4}  {sil_s:>14.4f}  {sil_sk:>14.4f}  {ari_k:>8.4f}")

    section("4. K-MEANS LIMITATIONS — NON-CONVEX SHAPES")
    X_moon, y_moon = make_moons(n_samples=300, noise=0.08, random_state=0)
    km_moon = KMeans(k=2, n_init=10, seed=0).fit(X_moon)
    ari_moon_km = adjusted_rand_score(y_moon, km_moon.labels_)
    print(f"Moons dataset (2 non-convex clusters):")
    print(f"  K-Means ARI: {ari_moon_km:.4f}  (poor — assumes convex clusters)")

    section("5. DBSCAN — DENSITY-BASED CLUSTERING")
    # Moons: DBSCAN should handle this perfectly
    dbscan_params = [
        (0.1, 5),
        (0.2, 5),
        (0.3, 5),
        (0.5, 5),
    ]
    print(f"Moons dataset — DBSCAN parameter sweep:")
    print(f"{'ε':>6}  {'MinPts':>8}  {'n_clusters':>12}  {'n_noise':>9}  {'ARI':>8}")
    print("-" * 48)
    for eps, min_pts in dbscan_params:
        db = DBSCAN(eps=eps, min_samples=min_pts).fit(X_moon)
        lbl = db.labels_
        n_clusters = len(set(lbl)) - (1 if -1 in lbl else 0)
        n_noise = (lbl == -1).sum()
        valid = lbl != -1
        ari_db = adjusted_rand_score(y_moon[valid], lbl[valid]) if valid.sum() > 10 else 0.0
        print(f"{eps:>6.1f}  {min_pts:>8}  {n_clusters:>12}  {n_noise:>9}  {ari_db:>8.4f}")

    # Best DBSCAN on moons
    db_best = DBSCAN(eps=0.2, min_samples=5).fit(X_moon)
    print(f"\n  Best DBSCAN (ε=0.2) ARI: {adjusted_rand_score(y_moon[db_best.labels_!=-1], db_best.labels_[db_best.labels_!=-1]):.4f}")

    section("6. DBSCAN OUTLIER DETECTION")
    rng = np.random.default_rng(7)
    X_clean, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=0)
    X_outliers = rng.uniform(-5, 5, (20, 2))
    X_od = np.vstack([X_clean, X_outliers])
    X_od = StandardScaler().fit_transform(X_od)

    db_od = DBSCAN(eps=0.4, min_samples=8).fit(X_od)
    n_outliers_detected = (db_od.labels_ == -1).sum()
    true_outlier_recall = (db_od.labels_[-20:] == -1).mean()

    print(f"Injected {20} outliers into 200-point dataset:")
    print(f"  DBSCAN detected outliers: {n_outliers_detected}")
    print(f"  Recall of true outliers:  {true_outlier_recall:.2f}")
    print(f"  n_clusters found: {len(set(db_od.labels_)) - 1}")

    section("7. K-MEANS++ VS RANDOM INIT")
    X_chal, _ = make_blobs(n_samples=500, centers=6, cluster_std=1.2, random_state=5)
    X_chal = StandardScaler().fit_transform(X_chal)

    inertias_pp = [KMeans(k=6, n_init=1, seed=s).fit(X_chal).inertia_ for s in range(20)]
    sk_rand = SKLearnKMeans(n_clusters=6, init="random", n_init=1, random_state=42).fit(X_chal)
    sk_pp = SKLearnKMeans(n_clusters=6, init="k-means++", n_init=1, random_state=42).fit(X_chal)

    print(f"K-Means++ (20 single runs):")
    print(f"  Mean inertia: {np.mean(inertias_pp):.2f}  Std: {np.std(inertias_pp):.2f}")
    print(f"  sklearn k-means++ n_init=1: {sk_pp.inertia_:.2f}")
    print(f"  sklearn random init  n_init=1: {sk_rand.inertia_:.2f}")


if __name__ == "__main__":
    main()
