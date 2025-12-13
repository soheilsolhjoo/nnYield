# convexity_numpy.py
# Simple convexity analysis module using only NumPy.
# Includes Hessian (finite differences), curvature diagnostics, Jensen convexity,
# and plotting utilities.

import numpy as np
import matplotlib.pyplot as plt
import os

# Example function: Von Mises (plane stress form) -----------------------------
def von_mises_func(x):
    x = np.array(x, dtype=np.float64)
    s11, s22, s12 = x[..., 0], x[..., 1], x[..., 2]
    term = s11**2 + s22**2 - s11*s22 + 3.0*s12**2
    return np.sqrt(np.maximum(term, 1e-12))

# Full Hessian via finite differences ----------------------------------------
def full_hessian(func, x, eps=1e-4):
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

# Principal curvature ---------------------------------------------------------
def principal_curvature(func, X):
    out = []
    for x in X:
        H = full_hessian(func, x)
        eigs = np.linalg.eigvalsh(H)
        out.append(eigs[0])
    return np.array(out)

# HVP curvature ---------------------------------------------------------------
def hvp_curvature(func, X, num_dirs=8):
    rng = np.random.default_rng(0)
    results = []
    for x in X:
        H = full_hessian(func, x)
        curv = []
        for _ in range(num_dirs):
            v = rng.standard_normal(3); v /= np.linalg.norm(v)
            curv.append(float(v @ H @ v))
        results.append(curv)
    return np.array(results)

# Directional curvature -------------------------------------------------------
def directional_curvature(func, X, num_dirs=8, h=1e-3):
    rng = np.random.default_rng(1)
    results = []
    for x in X:
        curvs = []
        for _ in range(num_dirs):
            v = rng.standard_normal(3); v /= np.linalg.norm(v)
            f_plus  = func(x + h*v)
            f_minus = func(x - h*v)
            f0      = func(x)
            curvs.append((f_plus - 2*f0 + f_minus) / (h*h))
        results.append(curvs)
    return np.array(results)

# Multi-scale curvature -------------------------------------------------------
def multiscale_curvature(func, X, num_dirs=4, hs=(1e-1,5e-2,1e-2,5e-3,1e-3)):
    rng = np.random.default_rng(2)
    results = []
    for x in X:
        for _ in range(num_dirs):
            v = rng.standard_normal(3); v /= np.linalg.norm(v)
            out = []
            for h in hs:
                f_plus = func(x + h*v)
                f_minus = func(x - h*v)
                f0 = func(x)
                out.append((f_plus - 2*f0 + f_minus)/(h*h))
            results.append(out)
    return np.array(results)

# Jensen convexity ------------------------------------------------------------
def segment_convexity(func, X, num_pairs=200, num_lam=4):
    rng = np.random.default_rng(3)
    violations = []
    for _ in range(num_pairs):
        i, j = rng.integers(0, len(X), size=2)
        x, y = X[i], X[j]
        for _ in range(num_lam):
            l = rng.uniform(0.05, 0.95)
            z = l*x + (1-l)*y
            fz = func(z)
            rhs = l*func(x) + (1-l)*func(y)
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
def run_full_comparison(func, outdir="convexity_output", N=5000):
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

    neg_frac = (pcurv < -1e-8).mean()
    print(f"λ_min negatives (tolerance -1e-8): {neg_frac:.4f}")
    print(f"Directional curvature median: {np.median(dcurv):.6f}")
    print(f"Jensen violation max: {jcv.max():.6e}")
    print(f"All plots saved in {outdir}")

if __name__ == "__main__":
    run_full_comparison(von_mises_func)
