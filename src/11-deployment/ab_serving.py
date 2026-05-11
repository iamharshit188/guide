"""
A/B model serving: traffic splitting, canary deployments, rollback triggers,
shadow mode, and latency-aware routing.
Covers: weighted random routing, sticky sessions, canary ramp-up schedule,
        metric-based rollback, shadow mode (log-only new model).
No external deps required.
"""

import time
import threading
import random
from collections import defaultdict, deque

rng_obj = random.Random(42)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Dummy Models ──────────────────────────────────────────────────
class ModelV1:
    """Stable production model. Accurate, slower."""
    def predict(self, X):
        time.sleep(0.005)   # simulate 5ms inference
        return [int(sum(x) > 0) for x in X]

    def predict_proba(self, X):
        time.sleep(0.005)
        return [[0.3, 0.7] if sum(x) > 0 else [0.7, 0.3] for x in X]


class ModelV2:
    """New candidate model. Faster but slightly less accurate."""
    def predict(self, X):
        time.sleep(0.002)   # 2ms — faster
        return [int(sum(x) > 0.5) for x in X]   # slightly different threshold

    def predict_proba(self, X):
        time.sleep(0.002)
        return [[0.25, 0.75] if sum(x) > 0.5 else [0.75, 0.25] for x in X]


# ── Metrics Collector ─────────────────────────────────────────────
class ModelMetrics:
    def __init__(self, window: int = 500):
        self._lock     = threading.Lock()
        self._latencies = deque(maxlen=window)
        self._errors    = deque(maxlen=window)
        self._requests  = 0

    def record(self, latency_ms: float, is_error: bool = False):
        with self._lock:
            self._latencies.append(latency_ms)
            self._errors.append(int(is_error))
            self._requests += 1

    def p50(self) -> float:
        with self._lock:
            lats = sorted(self._latencies)
            return lats[len(lats) // 2] if lats else 0.0

    def p95(self) -> float:
        with self._lock:
            lats = sorted(self._latencies)
            return lats[int(len(lats) * 0.95)] if lats else 0.0

    def error_rate(self) -> float:
        with self._lock:
            return sum(self._errors) / len(self._errors) if self._errors else 0.0

    def total_requests(self) -> int:
        with self._lock:
            return self._requests

    def summary(self) -> dict:
        return {
            "requests":   self.total_requests(),
            "p50_ms":     self.p50(),
            "p95_ms":     self.p95(),
            "error_rate": self.error_rate(),
        }


# ── A/B Router ───────────────────────────────────────────────────
class ABRouter:
    """
    Routes traffic between model versions using weighted random selection.
    Thread-safe via RLock.
    """

    def __init__(self):
        self._lock    = threading.RLock()
        self._models  = {}          # version → model
        self._weights = {}          # version → weight (sum to 1)
        self._metrics = {}          # version → ModelMetrics

    def register(self, version: str, model, weight: float):
        with self._lock:
            self._models[version]  = model
            self._metrics[version] = ModelMetrics()
        self.set_weight(version, weight)

    def set_weight(self, version: str, weight: float):
        with self._lock:
            self._weights[version] = weight
            total = sum(self._weights.values())
            self._weights = {k: v/total for k, v in self._weights.items()}

    def route(self) -> str:
        r = rng_obj.random()
        cumulative = 0.0
        with self._lock:
            items = list(self._weights.items())
        for version, weight in items:
            cumulative += weight
            if r < cumulative:
                return version
        return items[-1][0]

    def predict(self, X):
        version = self.route()
        model   = self._models[version]
        metrics = self._metrics[version]

        start = time.perf_counter()
        try:
            result = model.predict(X)
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record(latency_ms, is_error=False)
            return result, version
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record(latency_ms, is_error=True)
            raise

    def get_stats(self) -> dict:
        return {
            v: {"weight": self._weights.get(v, 0), **self._metrics[v].summary()}
            for v in self._models
        }


# ── Canary Controller ──────────────────────────────────────────────
class CanaryController:
    """
    Gradually ramps canary traffic from initial_pct to target_pct.
    Monitors metrics and rolls back if thresholds are violated.
    """

    def __init__(self, router: ABRouter,
                 stable_version: str,
                 canary_version: str,
                 initial_pct:    float = 0.01,
                 target_pct:     float = 1.00,
                 ramp_steps:     int   = 5,
                 max_error_rate: float = 0.05,
                 max_p95_ratio:  float = 1.50):

        self.router          = router
        self.stable          = stable_version
        self.canary          = canary_version
        self.initial_pct     = initial_pct
        self.target_pct      = target_pct
        self.ramp_steps      = ramp_steps
        self.max_error_rate  = max_error_rate
        self.max_p95_ratio   = max_p95_ratio
        self._current_pct    = 0.0
        self._step           = 0
        self._rolled_back    = False

    def start(self):
        self.router.set_weight(self.canary, self.initial_pct)
        self.router.set_weight(self.stable, 1.0 - self.initial_pct)
        self._current_pct = self.initial_pct
        print(f"  [Canary] Started at {self.initial_pct*100:.1f}%")

    def tick(self) -> bool:
        """
        Evaluate health and ramp up traffic if healthy.
        Returns False if rolled back, True if continuing.
        """
        if self._rolled_back:
            return False

        stats   = self.router.get_stats()
        s_stats = stats.get(self.stable, {})
        c_stats = stats.get(self.canary, {})

        # Rollback conditions
        if c_stats.get("requests", 0) > 10:   # wait for enough data
            c_err  = c_stats.get("error_rate", 0)
            c_p95  = c_stats.get("p95_ms", 0)
            s_p95  = s_stats.get("p95_ms", 1)

            if c_err > self.max_error_rate:
                print(f"  [Canary] ROLLBACK: canary error rate {c_err:.3f} "
                      f"> threshold {self.max_error_rate}")
                self._rollback()
                return False

            if s_p95 > 0 and c_p95 / s_p95 > self.max_p95_ratio:
                print(f"  [Canary] ROLLBACK: canary p95 {c_p95:.1f}ms "
                      f"is {c_p95/s_p95:.2f}x stable p95 {s_p95:.1f}ms")
                self._rollback()
                return False

        # Ramp up
        self._step += 1
        progress   = self._step / self.ramp_steps
        new_pct    = self.initial_pct + progress * (self.target_pct - self.initial_pct)
        new_pct    = min(new_pct, self.target_pct)
        self.router.set_weight(self.canary, new_pct)
        self.router.set_weight(self.stable, 1.0 - new_pct)
        self._current_pct = new_pct
        print(f"  [Canary] Step {self._step}/{self.ramp_steps}: "
              f"canary at {new_pct*100:.1f}%  |  "
              f"canary_err={c_stats.get('error_rate',0):.3f}  "
              f"canary_p95={c_stats.get('p95_ms',0):.1f}ms")

        if new_pct >= self.target_pct:
            print(f"  [Canary] SUCCESS: full migration to {self.canary}")
            return False

        return True

    def _rollback(self):
        self.router.set_weight(self.stable, 1.0)
        self.router.set_weight(self.canary, 0.0)
        self._rolled_back = True
        print(f"  [Canary] All traffic restored to {self.stable}")


# ── Shadow Mode ───────────────────────────────────────────────────
class ShadowRouter:
    """
    Routes all traffic to the stable model.
    Asynchronously runs the shadow model on a copy of each request.
    Shadow predictions are logged but never returned to clients.
    """

    def __init__(self, stable_model, shadow_model):
        self.stable  = stable_model
        self.shadow  = shadow_model
        self._log    = []
        self._lock   = threading.Lock()

    def predict(self, X):
        stable_result = self.stable.predict(X)

        # Fire-and-forget shadow evaluation
        thread = threading.Thread(target=self._shadow_call, args=(X, stable_result))
        thread.daemon = True
        thread.start()

        return stable_result

    def _shadow_call(self, X, stable_result):
        try:
            shadow_result = self.shadow.predict(X)
            agreement = sum(s == sh for s, sh in zip(stable_result, shadow_result)) / len(stable_result)
            with self._lock:
                self._log.append({"agreement": agreement})
        except Exception as e:
            with self._lock:
                self._log.append({"error": str(e)})

    def get_shadow_stats(self) -> dict:
        with self._lock:
            log = list(self._log)
        agreements = [e["agreement"] for e in log if "agreement" in e]
        errors     = [e for e in log if "error" in e]
        return {
            "total_shadow_calls": len(log),
            "mean_agreement":     sum(agreements) / len(agreements) if agreements else 0,
            "error_count":        len(errors),
        }


# ── Sticky Session (IP Hash) ──────────────────────────────────────
class StickyRouter:
    """Routes the same client_id to the same model version (session affinity)."""

    def __init__(self, models: dict):
        self._models  = models       # {"v1": model, "v2": model}
        self._map     = {}           # client_id → version
        self._lock    = threading.Lock()
        self._versions = list(models.keys())

    def get_model(self, client_id: str):
        with self._lock:
            if client_id not in self._map:
                # Use hash for deterministic assignment
                idx = hash(client_id) % len(self._versions)
                self._map[client_id] = self._versions[idx]
            return self._models[self._map[client_id]], self._map[client_id]

    def predict(self, client_id: str, X):
        model, version = self.get_model(client_id)
        return model.predict(X), version


def main():
    section("1. A/B ROUTER — WEIGHTED TRAFFIC SPLIT")
    router = ABRouter()
    v1, v2 = ModelV1(), ModelV2()
    router.register("v1", v1, weight=0.9)
    router.register("v2", v2, weight=0.1)

    print("  Sending 200 requests (90% → v1, 10% → v2)...")
    X = [[rng_obj.gauss(0, 1) for _ in range(5)] for _ in range(10)]

    for _ in range(200):
        try:
            _, version = router.predict(X)
        except Exception:
            pass

    stats = router.get_stats()
    for version, s in stats.items():
        print(f"  [{version}] weight={s['weight']:.2f}  requests={s['requests']}  "
              f"p50={s['p50_ms']:.2f}ms  p95={s['p95_ms']:.2f}ms  "
              f"err_rate={s['error_rate']:.3f}")

    section("2. CANARY DEPLOYMENT — RAMP-UP + HEALTH MONITORING")
    router2 = ABRouter()
    router2.register("v1_stable", ModelV1(), weight=1.0)
    router2.register("v2_canary", ModelV2(), weight=0.0)

    canary = CanaryController(
        router2,
        stable_version="v1_stable",
        canary_version="v2_canary",
        initial_pct=0.05,
        target_pct=1.00,
        ramp_steps=4,
        max_error_rate=0.10,
        max_p95_ratio=2.0,
    )
    canary.start()

    # Simulate traffic + canary ticks
    for tick in range(5):
        for _ in range(100):
            try:
                router2.predict([[rng_obj.gauss(0,1) for _ in range(5)]])
            except Exception:
                pass
        if not canary.tick():
            break

    print("\n  Final stats:")
    for version, s in router2.get_stats().items():
        print(f"  [{version}] weight={s['weight']:.2f}  requests={s['requests']}")

    section("3. SHADOW MODE")
    shadow_router = ShadowRouter(stable_model=ModelV1(), shadow_model=ModelV2())

    print("  Sending 50 requests through shadow router...")
    for _ in range(50):
        result = shadow_router.predict([[rng_obj.gauss(0,1) for _ in range(5)]])

    # Wait briefly for shadow threads to complete
    time.sleep(0.2)

    shadow_stats = shadow_router.get_shadow_stats()
    print(f"  Shadow calls: {shadow_stats['total_shadow_calls']}")
    print(f"  Mean agreement: {shadow_stats['mean_agreement']:.3f}")
    print(f"  Shadow errors:  {shadow_stats['error_count']}")
    print("  All client responses came from stable model only ✓")

    section("4. STICKY SESSION ROUTING")
    sticky = StickyRouter({"v1": ModelV1(), "v2": ModelV2()})

    client_ids = [f"user_{i}" for i in range(20)]
    assignments = {}
    for cid in client_ids:
        X = [[rng_obj.gauss(0,1) for _ in range(5)]]
        _, version = sticky.predict(cid, X)
        assignments[cid] = version

    from collections import Counter
    dist = Counter(assignments.values())
    print(f"  20 clients assigned: {dict(dist)}")
    print(f"  Consistent routing: same client always → same version")

    # Verify consistency
    for cid in client_ids[:5]:
        _, v1 = sticky.predict(cid, [[1,2,3,4,5]])
        _, v2 = sticky.predict(cid, [[1,2,3,4,5]])
        assert v1 == v2, f"Routing inconsistent for {cid}"
    print(f"  Verified: 5 clients routed consistently across 2 calls each ✓")

    section("5. ROLLBACK TRIGGER DEMO")
    router3 = ABRouter()

    class BrokenModel:
        def predict(self, X):
            raise RuntimeError("Model crashed")

    router3.register("v1_good", ModelV1(), weight=0.8)
    router3.register("v3_broken", BrokenModel(), weight=0.2)

    print("  Sending 50 requests to router with broken canary (20% weight)...")
    errors_v3 = 0
    for _ in range(50):
        try:
            router3.predict([[rng_obj.gauss(0,1) for _ in range(5)]])
        except RuntimeError:
            errors_v3 += 1

    stats3 = router3.get_stats()
    print(f"\n  v1_good:  {stats3['v1_good']['requests']} reqs, "
          f"err_rate={stats3['v1_good']['error_rate']:.3f}")
    print(f"  v3_broken:{stats3['v3_broken']['requests']} reqs, "
          f"err_rate={stats3['v3_broken']['error_rate']:.3f}  "
          f"← exceeds threshold → would trigger rollback")

    print("""
  Rollback trigger logic:
    if canary.error_rate > 0.05:   → rollback
    if canary.p95 / stable.p95 > 1.5:  → rollback
    → set canary weight = 0, stable weight = 1
    → alert on-call engineer
""")


if __name__ == "__main__":
    main()
