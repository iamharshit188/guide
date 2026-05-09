import numpy as np
import time


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Analytical functions and their gradients ──────────────────

def f_quadratic(x):
    """f(x, y) = x^2 + 2y^2 + xy — has a unique global minimum at (0,0)."""
    return x[0]**2 + 2*x[1]**2 + x[0]*x[1]


def grad_quadratic(x):
    """∂f/∂x = 2x + y,  ∂f/∂y = 4y + x"""
    return np.array([2*x[0] + x[1], 4*x[1] + x[0]])


def f_rosenbrock(x):
    """Rosenbrock: f(x,y) = (a-x)^2 + b(y-x^2)^2, min at (a, a^2)=( 1,1)."""
    a, b = 1.0, 100.0
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2


def grad_rosenbrock(x):
    """Analytical gradient of Rosenbrock."""
    a, b = 1.0, 100.0
    dfdx = -2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2)
    dfdy = 2*b*(x[1] - x[0]**2)
    return np.array([dfdx, dfdy])


# ── Numerical gradient (central difference) ──────────────────

def numerical_gradient(f, x, eps=1e-5):
    """
    Central-difference approximation:
        ∂f/∂xᵢ ≈ (f(x + εeᵢ) - f(x - εeᵢ)) / 2ε
    Error: O(ε²)  [vs forward-difference O(ε)]
    """
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad


def gradient_check(f, grad_fn, x, eps=1e-5, tol=1e-4):
    """Compares analytical vs numerical gradient. Returns relative error."""
    g_analytical = grad_fn(x)
    g_numerical = numerical_gradient(f, x, eps)
    numerator = np.linalg.norm(g_analytical - g_numerical)
    denominator = np.linalg.norm(g_analytical) + np.linalg.norm(g_numerical) + 1e-10
    rel_error = numerator / denominator
    passed = rel_error < tol
    return rel_error, passed, g_analytical, g_numerical


# ── Gradient Descent ─────────────────────────────────────────

def gradient_descent(f, grad_fn, x0, lr, n_iters, verbose_every=None):
    """
    Standard batch gradient descent: xₜ₊₁ = xₜ - η ∇f(xₜ)
    Returns: list of (iteration, x, f(x), ||∇f||)
    """
    x = x0.copy().astype(float)
    history = []
    for t in range(n_iters):
        loss = f(x)
        grad = grad_fn(x)
        grad_norm = np.linalg.norm(grad)
        history.append((t, x.copy(), loss, grad_norm))
        if verbose_every and t % verbose_every == 0:
            print(f"  iter {t:4d}: loss={loss:.6f}, ||grad||={grad_norm:.6f}, x={x.round(4)}")
        if grad_norm < 1e-8:
            break
        x -= lr * grad
    return history


def gradient_descent_momentum(f, grad_fn, x0, lr, momentum, n_iters):
    """Heavy-ball method: v = μv + ∇f, x = x - η*v"""
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    history = []
    for t in range(n_iters):
        loss = f(x)
        grad = grad_fn(x)
        v = momentum * v + grad
        x -= lr * v
        history.append((t, x.copy(), loss, np.linalg.norm(grad)))
    return history


# ── Chain rule demonstration ─────────────────────────────────

def chain_rule_demo():
    section("4. CHAIN RULE DEMONSTRATION")
    print("Function: h(x) = sigmoid(x^2)  =  1 / (1 + exp(-x^2))")
    print("Chain rule: dh/dx = sigmoid(x^2) * (1 - sigmoid(x^2)) * 2x")

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def h(x_val):
        return sigmoid(x_val**2)

    def dh_dx(x_val):
        z = x_val**2
        dsig_dz = sigmoid(z) * (1 - sigmoid(z))   # ∂sigmoid/∂z
        dz_dx = 2 * x_val                           # ∂z/∂x
        return dsig_dz * dz_dx                       # chain rule

    test_x = np.array([0.5])
    num_grad = numerical_gradient(lambda x: h(x[0]), test_x)[0]
    ana_grad = dh_dx(test_x[0])

    print(f"\nx = 0.5")
    print(f"  Analytical dh/dx = {ana_grad:.8f}")
    print(f"  Numerical  dh/dx = {num_grad:.8f}")
    print(f"  Relative error   = {abs(ana_grad - num_grad) / (abs(ana_grad) + 1e-10):.2e}")

    # Multi-step chain rule: f(g(h(x))) = (x^3 + 1)^2 at x=2
    # h(x) = x^3 + 1
    # g(u) = u^2
    # f(x) = g(h(x))
    x = 2.0
    h_x = x**3 + 1        # 9
    g_h = h_x**2          # 81
    dg_dh = 2 * h_x       # ∂g/∂h = 18
    dh_dx_val = 3 * x**2  # ∂h/∂x = 12
    df_dx = dg_dh * dh_dx_val  # 18 * 12 = 216

    print(f"\nf(x) = (x^3+1)^2 at x=2:")
    print(f"  f(2) = {g_h}")
    print(f"  df/dx = 2*(x^3+1) * 3x^2 = {df_dx}  (chain rule)")

    def composite(t):
        return (t**3 + 1)**2

    test_t = np.array([2.0])
    num_check = numerical_gradient(composite, test_t)[0]
    print(f"  Numerical check: {num_check:.4f}")


# ── Partial derivative visualization ─────────────────────────

def partial_derivative_demo():
    section("5. PARTIAL DERIVATIVES — NUMERICAL VERIFICATION")

    def f(xy):
        x, y = xy
        return x**2 * y + 3 * x * y**2

    # Analytical: ∂f/∂x = 2xy + 3y^2,  ∂f/∂y = x^2 + 6xy
    def grad_f(xy):
        x, y = xy
        return np.array([2*x*y + 3*y**2, x**2 + 6*x*y])

    test_points = [np.array([1.0, 2.0]), np.array([-1.0, 3.0]), np.array([2.0, -1.0])]

    print(f"f(x,y) = x²y + 3xy²")
    print(f"∂f/∂x = 2xy + 3y²  |  ∂f/∂y = x² + 6xy\n")
    print(f"{'Point':20s} {'Analytical':25s} {'Numerical':25s} {'Error':10s}")
    print("-" * 80)

    for pt in test_points:
        rel_err, passed, g_ana, g_num = gradient_check(f, grad_f, pt)
        print(f"({pt[0]:4.1f},{pt[1]:4.1f})           "
              f"{str(g_ana.round(4)):25s} {str(g_num.round(4)):25s} "
              f"{rel_err:.2e}  {'✓' if passed else '✗'}")


def main():
    section("1. GRADIENT CHECK — QUADRATIC FUNCTION")
    print("f(x, y) = x² + 2y² + xy")
    print("∂f/∂x = 2x + y  |  ∂f/∂y = 4y + x\n")

    test_points = [np.array([1.0, 2.0]), np.array([-3.0, 5.0]), np.array([0.0, 0.0])]
    for x in test_points:
        rel_err, passed, g_ana, g_num = gradient_check(f_quadratic, grad_quadratic, x)
        print(f"x={x}: rel_error={rel_err:.2e}  {'PASS' if passed else 'FAIL'}")
        print(f"  analytical: {g_ana.round(6)}")
        print(f"  numerical:  {g_num.round(6)}")

    section("2. GRADIENT DESCENT — QUADRATIC (CONVEX)")
    print("Minimizing f(x,y) = x² + 2y² + xy  (minimum at origin)")

    x0 = np.array([5.0, 5.0])
    history = gradient_descent(f_quadratic, grad_quadratic, x0,
                               lr=0.1, n_iters=100, verbose_every=10)

    final = history[-1]
    print(f"\nFinal: iter={final[0]}, x={final[1].round(6)}, loss={final[2]:.8f}")
    print(f"Expected minimum: x=[0, 0], f=0")

    # Convergence analysis: loss vs iteration
    losses = [h[2] for h in history]
    print(f"\nLoss trajectory:")
    for i in [0, 10, 25, 50, 99]:
        if i < len(history):
            print(f"  iter {i:3d}: {history[i][2]:.8f}")

    section("3. GRADIENT DESCENT — ROSENBROCK (NON-CONVEX)")
    print("f(x,y) = (1-x)² + 100(y-x²)²  — famous banana-shaped valley")
    print("Global minimum at (1, 1)")

    x0 = np.array([-1.0, 1.0])

    # Plain GD with small lr
    print("\nPlain GD (lr=0.001, 2000 iters):")
    h_plain = gradient_descent(f_rosenbrock, grad_rosenbrock, x0,
                               lr=0.001, n_iters=2000, verbose_every=500)
    print(f"  Final x: {h_plain[-1][1].round(4)}, loss: {h_plain[-1][2]:.6f}")

    # Momentum helps on ill-conditioned surfaces
    print("\nGD + Momentum (lr=0.001, μ=0.9, 2000 iters):")
    h_mom = gradient_descent_momentum(f_rosenbrock, grad_rosenbrock, x0,
                                      lr=0.001, momentum=0.9, n_iters=2000)
    print(f"  Final x: {h_mom[-1][1].round(4)}, loss: {h_mom[-1][2]:.6f}")

    chain_rule_demo()
    partial_derivative_demo()

    section("6. LEARNING RATE SENSITIVITY")
    print("Quadratic f(x,y) = x² + 2y², starting from [5, 5]")

    def f_simple(x):
        return x[0]**2 + 2*x[1]**2

    def grad_simple(x):
        return np.array([2*x[0], 4*x[1]])

    x0 = np.array([5.0, 5.0])
    lr_configs = [0.01, 0.1, 0.24, 0.5]  # last two: near-diverging, diverging

    for lr in lr_configs:
        h = gradient_descent(f_simple, grad_simple, x0, lr=lr, n_iters=50)
        last_loss = h[-1][2]
        converged = last_loss < 1e-6
        status = "CONVERGED" if converged else ("DIVERGED" if last_loss > 1e6 else "SLOW")
        print(f"  lr={lr:.3f}: final_loss={last_loss:.4e}  [{status}]")


if __name__ == "__main__":
    main()
