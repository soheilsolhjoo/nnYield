# generic_convexity.py
# Fully ready-to-run convexity analysis module with NumPy / TensorFlow / JAX backends.
# Includes Hessian, HVP, directional curvature, multiscale curvature, Jensen convexity,
# plotting utilities, and a wrapper to run all analyses without TF retracing warnings.

import numpy as np
import matplotlib.pyplot as plt
import os

# Optional backends -----------------------------------------------------------
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Backend registry ------------------------------------------------------------
BACKENDS = {}
if TF_AVAILABLE: BACKENDS['tf'] = 'tf'
if JAX_AVAILABLE: BACKENDS['jax'] = 'jax'
BACKENDS['numpy'] = 'numpy'

ACTIVE_BACKEND = 'numpy'

def set_backend(name):
    global ACTIVE_BACKEND
    if name not in BACKENDS:
        print(f"[generic_convexity] Backend '{name}' not available. Falling back to NumPy.")
        ACTIVE_BACKEND = 'numpy'
    else:
        ACTIVE_BACKEND = name
    print(f"[generic_convexity] Using backend: {ACTIVE_BACKEND}")

# Backend-aware von Mises -----------------------------------------------------
def von_mises_func(x):
    if ACTIVE_BACKEND == 'tf' and TF_AVAILABLE:
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        s11, s22, s12 = x[..., 0], x[..., 1], x[..., 2]
        term = s11**2 + s22**2 - s11*s22 + 3.0*s12**2
        return tf.sqrt(tf.maximum(term, 1e-12))
    elif ACTIVE_BACKEND == 'jax' and JAX_AVAILABLE:
        x = jnp.array(x, dtype=jnp.float64)
        s11, s22, s12 = x[..., 0], x[..., 1], x[..., 2]
        term = s11**2 + s22**2 - s11*s22 + 3.0*s12**2
        return jnp.sqrt(jnp.maximum(term, 1e-12))
    else:
        x = np.array(x, dtype=np.float64)
        s11, s22, s12 = x[..., 0], x[..., 1], x[..., 2]
        term = s11**2 + s22**2 - s11*s22 + 3.0*s12**2
        return np.sqrt(np.maximum(term, 1e-12))

# Generic evaluation ----------------------------------------------------------
def evaluate_func(func, x):
    if ACTIVE_BACKEND == 'tf' and TF_AVAILABLE:
        return func(tf.convert_to_tensor(x, dtype=tf.float64)).numpy()
    elif ACTIVE_BACKEND == 'jax' and JAX_AVAILABLE:
        return np.array(func(jnp.array(x)))
    else:
        return func(np.array(x))

# TensorFlow compiled helpers (defined once to avoid retracing) ---------------
if TF_AVAILABLE:
    @tf.function(reduce_retracing=True)
    def _tf_hessian_compiled(func, x_tf):
        # x_tf: shape (3,), dtype float64
        with tf.GradientTape() as g2:
            g2.watch(x_tf)
            with tf.GradientTape() as g1:
                g1.watch(x_tf)
                f = tf.squeeze(func(x_tf))
            grad_val = g1.gradient(f, x_tf)
        return g2.jacobian(grad_val, x_tf)

    @tf.function(reduce_retracing=True)
    def _tf_hvp_compiled(func, x_tf, v_tf):
        # Returns v^T H(x) v; x_tf, v_tf shape (3,), dtype float64
        with tf.GradientTape() as g2:
            g2.watch(x_tf)
            with tf.GradientTape() as g1:
                g1.watch(x_tf)
                f = tf.squeeze(func(x_tf))
            grad_val = g1.gradient(f, x_tf)           # ∇f(x)
            dot = tf.tensordot(grad_val, v_tf, axes=1)  # v·∇f(x)
        hvp = g2.gradient(dot, x_tf)                  # H(x)v
        return tf.tensordot(hvp, v_tf, axes=1)        # v^T H v

# Unified Hessian -------------------------------------------------------------
def full_hessian(func, x):
    if ACTIVE_BACKEND == 'tf' and TF_AVAILABLE:
        x_tf = tf.convert_to_tensor(x, dtype=tf.float64)
        H = _tf_hessian_compiled(func, x_tf)
        return H.numpy()

    elif ACTIVE_BACKEND == 'jax' and JAX_AVAILABLE:
        return np.array(jax.hessian(func)(jnp.array(x)))

    else:  # NumPy finite-difference
        eps = 1e-4
        n = x.shape[-1]
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ei = np.zeros_like(x); ej = np.zeros_like(x)
                ei[i] = eps; ej[j] = eps
                fpp = func(x + ei + ej)
                fpm = func(x + ei - ej)
                fmp = func(x - ei + ej)
                fmm = func(x - ei - ej)
                H[i, j] = (fpp - fpm - fmp + fmm) / (4*eps*eps)
        return 0.5 * (H + H.T)

# Unified HVP curvature -------------------------------------------------------
def hvp_curvature(func, X, num_dirs=8, seed=None):
    rng = np.random.default_rng(seed)
    results = []
    for x in X:
        curv = []
        if ACTIVE_BACKEND == 'tf' and TF_AVAILABLE:
            x_tf = tf.convert_to_tensor(x, dtype=tf.float64)
            for _ in range(num_dirs):
                v = rng.standard_normal(x.shape[0]); v /= np.linalg.norm(v)
                v_tf = tf.convert_to_tensor(v, dtype=tf.float64)
                val = _tf_hvp_compiled(func, x_tf, v_tf)
                curv.append(float(val.numpy()))

        elif ACTIVE_BACKEND == 'jax' and JAX_AVAILABLE:
            xj = jnp.array(x)
            for _ in range(num_dirs):
                v = rng.standard_normal(x.shape[0]); v /= np.linalg.norm(v)
                _, Hv = jax.jvp(jax.grad(func), (xj,), (jnp.array(v),))
                curv.append(float(jnp.dot(Hv, v)))

        else:  # NumPy fallback
            H = full_hessian(func, x)
            for _ in range(num_dirs):
                v = rng.standard_normal(x.shape[0]); v /= np.linalg.norm(v)
                curv.append(float(v @ H @ v))

        results.append(curv)
    return np.array(results)

# Principal curvature ---------------------------------------------------------
def principal_curvature(func, X):
    return np.array([np.linalg.eigvalsh(full_hessian(func, x))[0] for x in X])

# Directional curvature -------------------------------------------------------
def directional_curvature(func, X, num_dirs=8, h=1e-3):
    rng = np.random.default_rng(0)
    results = []
    for x in X:
        curvs = []
        for _ in range(num_dirs):
            v = rng.standard_normal(3); v /= np.linalg.norm(v)
            f_plus  = evaluate_func(func, x + h*v)
            f_minus = evaluate_func(func, x - h*v)
            f0      = evaluate_func(func, x)
            curvs.append((f_plus - 2*f0 + f_minus) / (h*h))
        results.append(curvs)
    return np.array(results)

# Multi-scale curvature -------------------------------------------------------
def multiscale_curvature(func, X, num_dirs=4, hs=(1e-1,5e-2,1e-2,5e-3,1e-3)):
    rng = np.random.default_rng(1)
    results = []
    for x in X:
        for _ in range(num_dirs):
            v = rng.standard_normal(3); v /= np.linalg.norm(v)
            out = []
            for h in hs:
                f_plus = evaluate_func(func, x + h*v)
                f_minus = evaluate_func(func, x - h*v)
                f0 = evaluate_func(func, x)
                out.append((f_plus - 2*f0 + f_minus)/(h*h))
            results.append(out)
    return np.array(results)

# Jensen convexity ------------------------------------------------------------
def segment_convexity(func, X, num_pairs=200, num_lam=4):
    rng = np.random.default_rng(2)
    violations = []
    for _ in range(num_pairs):
        i, j = rng.integers(0, len(X), size=2)
        x, y = X[i], X[j]
        for _ in range(num_lam):
            l = rng.uniform(0.05, 0.95)
            z = l*x + (1-l)*y
            fz = evaluate_func(func, z)
            rhs = l*evaluate_func(func, x) + (1-l)*evaluate_func(func, y)
            violations.append(fz - rhs)
    return np.array(violations)

# Plotting --------------------------------------------------------------------
def plot_histogram_min_eigs(values, path):
    plt.figure(figsize=(7,5))
    plt.hist(values, bins=50, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--')
    plt.yscale('log')
    plt.title("Minimum Hessian Eigenvalues")
    plt.xlabel("λ_min"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path); plt.close()

def plot_stability_slice(func, path):
    theta = np.linspace(0, 2*np.pi, 300)
    X = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
    vals = principal_curvature(func, X)
    plt.figure(figsize=(8,4))
    plt.plot(theta/np.pi, vals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Principal curvature along s12=0 unit circle")
    plt.xlabel("θ / π"); plt.ylabel("λ_min(H)")
    plt.tight_layout()
    plt.savefig(path); plt.close()

# Wrapper run -----------------------------------------------------------------
def run_full_comparison(func, outdir="convexity_output", N=2000):
    os.makedirs(outdir, exist_ok=True)

    X = np.random.randn(N,3)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    print("Running principal curvature...")
    pcurv = principal_curvature(func, X)
    plot_histogram_min_eigs(pcurv, os.path.join(outdir, 'hist_min_eigs.png'))

    print("Running directional curvature...")
    dcurv = directional_curvature(func, X)

    print("Running multiscale curvature...")
    mcurv = multiscale_curvature(func, X)

    print("Running Jensen convexity...")
    jcv = segment_convexity(func, X)

    print("Running slice plot...")
    plot_stability_slice(func, os.path.join(outdir, 'slice.png'))

    # Simple textual summary
    neg_frac = (pcurv < -1e-8).mean()
    print(f"λ_min negatives (tolerance -1e-8): {neg_frac:.4f}")
    print(f"Directional curvature median: {np.median(dcurv):.6f}")
    print(f"Jensen violation max: {jcv.max():.6e}")
    print(f"All plots saved in {outdir}")

if __name__ == "__main__":
    # Choose backend: 'numpy' is always available; 'tf' or 'jax' if installed.
    set_backend('numpy')  # change to 'numpy' or 'jax' as desired
    run_full_comparison(von_mises_func)
