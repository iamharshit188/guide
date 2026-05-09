"""
Optimizers from scratch — SGD, Momentum, Nesterov, RMSProp, Adam, AdamW.
Includes LR scheduling (step, cosine, warmup+cosine) and convergence benchmarks.
pip install numpy
"""

import numpy as np
import math


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ══════════════════════════════════════════════════════════════
# OPTIMIZER BASE & IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════

class Optimizer:
    """Base class — subclasses implement step()."""

    def __init__(self, params: dict, lr: float):
        # params: {name: np.ndarray} — modified in-place
        self.params = params
        self.lr = lr
        self.t = 0          # step counter

    def step(self, grads: dict):
        """grads: {name: np.ndarray} matching params."""
        raise NotImplementedError

    def zero_grad(self):
        self.t += 1


class SGD(Optimizer):
    """Vanilla stochastic gradient descent."""

    def step(self, grads):
        self.zero_grad()
        for name, g in grads.items():
            self.params[name] -= self.lr * g


class SGDMomentum(Optimizer):
    """
    Heavy-ball momentum:
      v = mu * v - lr * g
      theta += v
    """

    def __init__(self, params, lr, momentum=0.9):
        super().__init__(params, lr)
        self.mu = momentum
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.zero_grad()
        for name, g in grads.items():
            self.v[name] = self.mu * self.v[name] - self.lr * g
            self.params[name] += self.v[name]


class NesterovMomentum(Optimizer):
    """
    Nesterov accelerated gradient:
      v_prev = v
      v = mu * v - lr * g(theta + mu * v)
      theta += -mu * v_prev + (1 + mu) * v
    Equivalent to computing gradient at the lookahead position.
    """

    def __init__(self, params, lr, momentum=0.9):
        super().__init__(params, lr)
        self.mu = momentum
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.zero_grad()
        for name, g in grads.items():
            v_prev = self.v[name].copy()
            self.v[name] = self.mu * self.v[name] - self.lr * g
            self.params[name] += -self.mu * v_prev + (1 + self.mu) * self.v[name]


class RMSProp(Optimizer):
    """
    Adaptive per-parameter LR using running 2nd-moment estimate:
      s = beta * s + (1 - beta) * g^2
      theta -= lr / sqrt(s + eps) * g
    """

    def __init__(self, params, lr=1e-3, beta=0.99, eps=1e-8):
        super().__init__(params, lr)
        self.beta = beta
        self.eps = eps
        self.s = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.zero_grad()
        for name, g in grads.items():
            self.s[name] = self.beta * self.s[name] + (1 - self.beta) * g**2
            self.params[name] -= self.lr / (np.sqrt(self.s[name]) + self.eps) * g


class Adam(Optimizer):
    """
    Adaptive Moment Estimation:
      m = beta1 * m + (1 - beta1) * g          (1st moment)
      v = beta2 * v + (1 - beta2) * g^2        (2nd moment)
      m_hat = m / (1 - beta1^t)                (bias correction)
      v_hat = v / (1 - beta2^t)
      theta -= lr / (sqrt(v_hat) + eps) * m_hat
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.zero_grad()
        t = self.t
        bc1 = 1 - self.beta1**t   # bias correction factor
        bc2 = 1 - self.beta2**t
        for name, g in grads.items():
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g**2
            m_hat = self.m[name] / bc1
            v_hat = self.v[name] / bc2
            self.params[name] -= self.lr / (np.sqrt(v_hat) + self.eps) * m_hat


class AdamW(Optimizer):
    """
    Adam with decoupled weight decay:
      theta -= lr * lambda_wd * theta   (weight decay applied directly)
      then standard Adam gradient update

    Decoupling ensures weight decay is not scaled by adaptive LR denominator.
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.zero_grad()
        t = self.t
        bc1 = 1 - self.beta1**t
        bc2 = 1 - self.beta2**t
        for name, g in grads.items():
            # Weight decay — applied before gradient update
            self.params[name] -= self.lr * self.wd * self.params[name]
            # Adam update
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g**2
            m_hat = self.m[name] / bc1
            v_hat = self.v[name] / bc2
            self.params[name] -= self.lr / (np.sqrt(v_hat) + self.eps) * m_hat


# ══════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULERS
# ══════════════════════════════════════════════════════════════

class StepDecay:
    """Multiply LR by gamma every step_size epochs."""

    def __init__(self, base_lr, gamma=0.1, step_size=30):
        self.base_lr = base_lr
        self.gamma = gamma
        self.step_size = step_size

    def get_lr(self, epoch):
        return self.base_lr * self.gamma ** (epoch // self.step_size)


class CosineAnnealing:
    """
    Cosine annealing without restarts:
      lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*t/T))
    """

    def __init__(self, lr_max, lr_min=0.0, T_max=100):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max = T_max

    def get_lr(self, t):
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + math.cos(math.pi * t / self.T_max)
        )


class WarmupCosine:
    """
    Linear warmup for T_warm steps, then cosine decay to lr_min.
    Standard schedule for transformer pre-training.
    """

    def __init__(self, lr_max, lr_min=0.0, T_warm=1000, T_total=10000):
        self.lr_max  = lr_max
        self.lr_min  = lr_min
        self.T_warm  = T_warm
        self.T_total = T_total

    def get_lr(self, t):
        if t <= self.T_warm:
            return self.lr_max * t / self.T_warm
        progress = (t - self.T_warm) / max(self.T_total - self.T_warm, 1)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + math.cos(math.pi * progress)
        )


class ReduceOnPlateau:
    """
    Halve LR after `patience` steps with no improvement.
    Monitors a scalar metric (e.g., validation loss).
    """

    def __init__(self, base_lr, factor=0.5, patience=10, min_lr=1e-6):
        self.lr = base_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self._best = float("inf")
        self._wait = 0

    def step(self, metric):
        if metric < self._best:
            self._best = metric
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.lr = max(self.lr * self.factor, self.min_lr)
                self._wait = 0
        return self.lr


# ══════════════════════════════════════════════════════════════
# TEST OBJECTIVES
# ══════════════════════════════════════════════════════════════

def rosenbrock(x, y):
    """f(x,y) = (1-x)^2 + 100(y-x^2)^2 — classic non-convex test."""
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2*(1 - x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

def quadratic(theta):
    """f(theta) = 0.5 * theta^T A theta — convex, exact optimum at 0."""
    A = np.diag([1.0, 10.0, 100.0])  # ill-conditioned (condition number 100)
    return 0.5 * theta @ A @ theta

def quadratic_grad(theta):
    A = np.diag([1.0, 10.0, 100.0])
    return A @ theta


# ══════════════════════════════════════════════════════════════
# CONVERGENCE BENCHMARK
# ══════════════════════════════════════════════════════════════

def run_optimizer(opt_class, opt_kwargs, grad_fn, loss_fn, init,
                  n_steps=500, scheduler=None):
    """Run optimizer for n_steps, return loss history."""
    params = {"theta": init.copy()}
    opt = opt_class(params, **opt_kwargs)
    history = []

    for step in range(n_steps):
        if scheduler is not None:
            opt.lr = scheduler.get_lr(step)
        g = grad_fn(params["theta"])
        grads = {"theta": g}
        opt.step(grads)
        loss = loss_fn(params["theta"])
        history.append(float(loss))

    return history, params["theta"].copy()


def benchmark_optimizers_quadratic():
    section("CONVERGENCE — ILL-CONDITIONED QUADRATIC (A=diag[1,10,100])")
    init = np.array([5.0, 5.0, 5.0])
    n = 300

    configs = [
        ("SGD lr=0.01",        SGD,          {"lr": 0.01}),
        ("SGD lr=0.1",         SGD,          {"lr": 0.1}),
        ("Momentum mu=0.9",    SGDMomentum,  {"lr": 0.01, "momentum": 0.9}),
        ("Nesterov mu=0.9",    NesterovMomentum, {"lr": 0.01, "momentum": 0.9}),
        ("RMSProp",            RMSProp,      {"lr": 0.1}),
        ("Adam",               Adam,         {"lr": 0.1}),
        ("AdamW wd=0.01",      AdamW,        {"lr": 0.1, "weight_decay": 0.01}),
    ]

    print(f"  {'Optimizer':22s}  {'Loss@50':>10s}  {'Loss@150':>10s}  "
          f"{'Loss@300':>10s}  {'Final θ norm':>12s}")
    print("  " + "-" * 70)

    for name, cls, kwargs in configs:
        history, final = run_optimizer(
            cls, kwargs,
            grad_fn=quadratic_grad,
            loss_fn=quadratic,
            init=init, n_steps=n,
        )
        l50  = history[49]  if len(history) > 49  else float("nan")
        l150 = history[149] if len(history) > 149 else float("nan")
        l300 = history[-1]
        fnorm = float(np.linalg.norm(final))
        print(f"  {name:22s}  {l50:>10.4f}  {l150:>10.4f}  "
              f"{l300:>10.6f}  {fnorm:>12.6f}")


def benchmark_optimizers_rosenbrock():
    section("CONVERGENCE — ROSENBROCK (non-convex, minimum at (1,1))")
    init = np.array([-1.0, 1.0])
    n = 2000

    configs = [
        ("SGD lr=1e-3",       SGD,          {"lr": 1e-3}),
        ("Momentum mu=0.9",   SGDMomentum,  {"lr": 1e-3, "momentum": 0.9}),
        ("Nesterov mu=0.9",   NesterovMomentum, {"lr": 1e-3, "momentum": 0.9}),
        ("RMSProp",           RMSProp,      {"lr": 1e-3}),
        ("Adam",              Adam,         {"lr": 1e-3}),
    ]

    def rosen_grad_vec(theta):
        return rosenbrock_grad(theta[0], theta[1])

    def rosen_loss_vec(theta):
        return rosenbrock(theta[0], theta[1])

    print(f"  {'Optimizer':22s}  {'Loss@500':>10s}  {'Loss@1000':>10s}  "
          f"{'Loss@2000':>10s}  {'Dist from (1,1)':>15s}")
    print("  " + "-" * 72)

    for name, cls, kwargs in configs:
        history, final = run_optimizer(
            cls, kwargs,
            grad_fn=rosen_grad_vec,
            loss_fn=rosen_loss_vec,
            init=init, n_steps=n,
        )
        l500  = history[499]  if len(history) > 499  else float("nan")
        l1000 = history[999]  if len(history) > 999  else float("nan")
        l2000 = history[-1]
        dist  = float(np.linalg.norm(final - np.array([1.0, 1.0])))
        print(f"  {name:22s}  {l500:>10.4f}  {l1000:>10.4f}  "
              f"{l2000:>10.6f}  {dist:>15.6f}")


# ══════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULER DEMO
# ══════════════════════════════════════════════════════════════

def demo_lr_schedules():
    section("LEARNING RATE SCHEDULES")
    T = 100

    schedules = {
        "StepDecay(γ=0.5,s=25)":   StepDecay(1.0, gamma=0.5, step_size=25),
        "CosineAnnealing(min=0)":   CosineAnnealing(lr_max=1.0, lr_min=0.0, T_max=T),
        "CosineAnnealing(min=0.1)": CosineAnnealing(lr_max=1.0, lr_min=0.1, T_max=T),
        "WarmupCosine(warm=10)":    WarmupCosine(lr_max=1.0, lr_min=0.0,
                                                  T_warm=10, T_total=T),
    }

    checkpoints = [0, 10, 25, 50, 75, 99]
    print(f"  {'Schedule':30s}", end="")
    for t in checkpoints:
        print(f"  t={t:3d}", end="")
    print()
    print("  " + "-" * 80)

    for name, sched in schedules.items():
        print(f"  {name:30s}", end="")
        for t in checkpoints:
            lr = sched.get_lr(t)
            print(f"  {lr:.3f}", end="")
        print()

    # ReduceOnPlateau demo
    print(f"\n  ReduceOnPlateau (factor=0.5, patience=5):")
    plateau = ReduceOnPlateau(base_lr=0.1, factor=0.5, patience=5, min_lr=1e-5)
    losses = [1.0, 0.9, 0.85, 0.85, 0.85, 0.85, 0.85,  # plateau for 5 steps
              0.84, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80]  # another plateau
    print(f"  {'Step':>5s}  {'Val Loss':>10s}  {'LR after step':>14s}")
    print("  " + "-" * 35)
    for i, loss in enumerate(losses):
        lr = plateau.step(loss)
        print(f"  {i:>5d}  {loss:>10.4f}  {lr:>14.6f}")


def demo_bias_correction():
    section("ADAM BIAS CORRECTION — WHY IT MATTERS")
    g_true = 1.0   # constant gradient for clarity
    beta1, beta2 = 0.9, 0.999

    m, v = 0.0, 0.0
    print(f"  {'Step':>5s}  {'m (raw)':>10s}  {'m_hat (corrected)':>18s}  "
          f"{'v (raw)':>10s}  {'v_hat':>10s}")
    print("  " + "-" * 60)
    for t in range(1, 11):
        m = beta1 * m + (1 - beta1) * g_true
        v = beta2 * v + (1 - beta2) * g_true**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        print(f"  {t:>5d}  {m:>10.6f}  {m_hat:>18.6f}  {v:>10.8f}  {v_hat:>10.6f}")

    print("""
  Without bias correction:
    At t=1: m = 0.1 (true gradient = 1.0) → step size 10x too small.
    At t=10: m ≈ 0.65 → still underestimating.
  With bias correction:
    At t=1: m_hat ≈ 1.0 (correct).
    Converges to true value within a few steps.
    """)


def demo_gradient_clipping():
    section("GRADIENT CLIPPING")
    print("""
  Problem: Exploding gradients in RNNs / deep networks.
  Gradient norms can grow to 1e5+ on long sequences.

  Gradient clipping by norm:
    if ||g||_2 > max_norm:
        g = g * (max_norm / ||g||_2)

  This preserves direction — only scales down magnitude.
  Clipping by value (clip each element): changes direction, less preferred.
    """)

    max_norm = 1.0
    rng = np.random.default_rng(0)
    for trial in range(5):
        g = rng.standard_normal(10) * (10 ** trial)
        norm_before = np.linalg.norm(g)
        if norm_before > max_norm:
            g_clipped = g * max_norm / norm_before
        else:
            g_clipped = g
        norm_after = np.linalg.norm(g_clipped)
        clipped = norm_before > max_norm
        print(f"  trial {trial}: ||g|| = {norm_before:.2f}  "
              f"→ {'CLIPPED' if clipped else 'no clip':7s}  "
              f"||g_clip|| = {norm_after:.4f}")


def demo_optimizer_internals():
    section("OPTIMIZER INTERNAL STATE — ADAM STEP TRACE")
    params = {"W": np.array([2.0, -1.0, 0.5])}
    opt = Adam(params, lr=0.1, beta1=0.9, beta2=0.999)
    grads_seq = [
        {"W": np.array([1.0,  0.5, -0.3])},
        {"W": np.array([0.8,  0.6, -0.4])},
        {"W": np.array([0.6,  0.7, -0.5])},
    ]
    print(f"  {'Step':>5s}  {'W[0]':>8s}  {'m[0]':>8s}  {'v[0]':>10s}  {'lr_eff[0]':>10s}")
    print("  " + "-" * 50)
    for i, g in enumerate(grads_seq):
        opt.step(g)
        t = opt.t
        m0 = opt.m["W"][0]
        v0 = opt.v["W"][0]
        m_hat = m0 / (1 - 0.9**t)
        v_hat = v0 / (1 - 0.999**t)
        lr_eff = 0.1 / (np.sqrt(v_hat) + 1e-8)
        print(f"  {t:>5d}  {params['W'][0]:>8.5f}  {m0:>8.5f}  "
              f"{v0:>10.7f}  {lr_eff:>10.5f}")


def main():
    benchmark_optimizers_quadratic()
    benchmark_optimizers_rosenbrock()
    demo_lr_schedules()
    demo_bias_correction()
    demo_gradient_clipping()
    demo_optimizer_internals()


if __name__ == "__main__":
    main()
