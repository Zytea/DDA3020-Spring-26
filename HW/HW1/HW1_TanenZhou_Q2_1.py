import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = ""

def f(x):
    return x[0] ** 2 + 5 * (x[1] ** 2)

def grad_f(x):
    return np.array([2 * x[0], 10 * x[1]], dtype=float)

def gradient_descent(eta, iters=50, x0=None):
    if x0 is None:
        x = np.array([5.0, 5.0], dtype=float)
    else:
        x = np.array(x0, dtype=float)
    traj = [x.copy()]
    for _ in range(iters):
        x -= eta * grad_f(x)
        traj.append(x.copy())
    return np.array(traj)

def converge_iters(eta, tol=1e-6, max_iters=10000, x0=None):
    if x0 is None:
        x = np.array([5.0, 5.0], dtype=float)
    else:
        x = np.array(x0, dtype=float)
    for t in range(int(max_iters)):
        if np.linalg.norm(grad_f(x)) < tol:
            return t, True, x.copy(), f(x)
        x -= eta * grad_f(x)
    return int(max_iters), False, x.copy(), f(x)

def plot_contour_with_traj(traj, path):
    xs = np.linspace(-6, 6, 400)
    ys = np.linspace(-6, 6, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = f(np.array([X, Y]))

    plt.figure(figsize=(6, 5))
    cs = plt.contour(X, Y, Z, levels=30)
    plt.clabel(cs, inline=1, fontsize=8, fmt="%.0f")
    plt.plot(traj[:, 0], traj[:, 1], marker="o", markersize=2, linewidth=1)
    plt.scatter([traj[0, 0]], [traj[0, 1]], marker="s", s=60)
    plt.scatter([traj[-1, 0]], [traj[-1, 1]], marker="*", s=120)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("GD trajectory (eta=0.1)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    # (a) 
    traj = gradient_descent(eta=0.1, iters=50)
    x_final = traj[-1]
    print("Part (a) eta=0.1, 50 iterations")
    print(f"Final x = [{x_final[0]:.10f}, {x_final[1]:.10f}]")
    print(f"Final f(x) = {f(x_final):.12e}")
    print()

    # (b) 
    plot_contour_with_traj(traj, OUT_DIR + "q2_1.png")
    print("Plot Saved!")
    print()

    # (c) 
    etas = [0.01, 0.1, 1.1]
    print("Part (c) convergence (tol on ||grad|| < 1e-6)")
    for eta in etas:
        it, ok, x_end, f_end = converge_iters(eta)
        if ok:
            print(
                f"eta={eta}: converged in {it} iterations, x≈[{x_end[0]:.3e}, {x_end[1]:.3e}], f≈{f_end:.3e}"
            )
        else:
            print(
                f"eta={eta}: diverged by iter {it}, x≈[{x_end[0]:.3e}, {x_end[1]:.3e}], f≈{f_end:.3e}"
            )

if __name__ == "__main__":
    main()

'''
Part (a) eta=0.1, 50 iterations
Final x = [0.0000713624, 0.0000000000]
Final f(x) = 5.092589940836e-09

Plot Saved!

Part (c) convergence (tol on ||grad|| < 1e-6)
eta=0.01: converged in 798 iterations, x≈[4.982e-07, 1.529e-36], f≈2.482e-13
eta=0.1: converged in 73 iterations, x≈[4.212e-07, 0.000e+00], f≈1.775e-13
eta=1.1: diverged by iter 10000, x≈[nan, nan], f≈nan

Saved: q2_1_eta_comparison.png
'''