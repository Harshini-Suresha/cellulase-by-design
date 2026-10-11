"""Matplotlib figures for the app (kept separate so they can be tested without Streamlit)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import rsm

BG, INK, ACC, HOT = "#FFE4EC", "#2A1520", "#1D4ED8", "#B8325F"


def _style(ax):
    ax.set_facecolor("#FFF6F9")
    for s in ax.spines.values():
        s.set_color("#EBC1D0")
    ax.tick_params(colors=INK, labelsize=8)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK); ax.title.set_color(INK)


def _fig(*a, **k):
    f, ax = plt.subplots(*a, **k)
    f.patch.set_facecolor(BG)
    return f, ax


def _surface(m, ca, cb, lim, n=121):
    g = np.linspace(-lim, lim, n)
    A, B = np.meshgrid(g, g)
    Z = rsm.predict(m, np.column_stack([A.ravel(), B.ravel()]))[0].reshape(A.shape)
    return A, B, Z


def contour(m, ca, cb, x, names, pt=None, stat=None, lim=1.414, region=True):
    f, ax = _fig(figsize=(5.6, 4.6))
    A, B, Z = _surface(m, ca, cb, lim)
    an, bn = ca.to_natural(A), cb.to_natural(B)
    cf = ax.contourf(an, bn, Z, 24, cmap="turbo", alpha=.9)
    cs = ax.contour(an, bn, Z, 10, colors=INK, linewidths=.5)
    ax.clabel(cs, fontsize=7, fmt="%.1f")
    if region:
        t = np.linspace(0, 2 * np.pi, 200)
        r = float(np.max(np.hypot(x[:, 0], x[:, 1])))
        ax.plot(ca.to_natural(r * np.cos(t)), cb.to_natural(r * np.sin(t)), "w--", lw=1)
    ax.scatter(ca.to_natural(x[:, 0]), cb.to_natural(x[:, 1]), s=34, c="white", edgecolor=INK, zorder=5, label="design points")
    if pt is not None:
        ax.scatter(*[c.to_natural(v) for c, v in zip((ca, cb), pt)], marker="*", s=240, c=HOT, edgecolor="white", zorder=6, label="selected")
    if stat is not None and np.abs(stat).max() <= lim:
        ax.scatter(*[c.to_natural(v) for c, v in zip((ca, cb), stat)], marker="X", s=80, c="white", edgecolor=INK, zorder=6, label="stationary point")
    ax.set_xlabel(names[0]); ax.set_ylabel(names[1]); ax.legend(fontsize=7, loc="upper left", framealpha=.85)
    f.colorbar(cf, ax=ax, shrink=.85).ax.tick_params(labelsize=7)
    _style(ax); f.tight_layout()
    return f


def surface3d(m, ca, cb, names, lim=1.414, elev=26, azim=-58):
    f = plt.figure(figsize=(5.6, 4.6)); f.patch.set_facecolor(BG)
    ax = f.add_subplot(projection="3d")
    A, B, Z = _surface(m, ca, cb, lim, 61)
    ax.plot_surface(ca.to_natural(A), cb.to_natural(B), Z, cmap="turbo", linewidth=0, antialiased=True, alpha=.95)
    ax.view_init(elev, azim)
    ax.set_xlabel(names[0], fontsize=8, color=INK); ax.set_ylabel(names[1], fontsize=8, color=INK)
    ax.set_zlabel("Response", fontsize=8, color=INK); ax.tick_params(labelsize=7, colors=INK)
    ax.set_facecolor(BG); f.tight_layout()
    return f


def diagnostics(m):
    f, axs = _fig(1, 3, figsize=(11, 3.4))
    r = m["stud"]
    axs[0].scatter(m["fitted"], r, c=ACC, s=26); axs[0].axhline(0, c=INK, lw=.8)
    for v in (-2, 2):
        axs[0].axhline(v, c=HOT, ls=":", lw=1)
    axs[0].set_xlabel("Fitted"); axs[0].set_ylabel("Studentised residual"); axs[0].set_title("Residuals vs fitted", fontsize=9)
    q = stats.norm.ppf((np.arange(1, len(r) + 1) - .5) / len(r))
    axs[1].scatter(q, np.sort(r), c=ACC, s=26)
    lim = max(abs(q).max(), abs(r).max()) * 1.05
    axs[1].plot([-lim, lim], [-lim, lim], c=HOT, lw=1)
    axs[1].set_xlabel("Normal quantile"); axs[1].set_ylabel("Ordered residual"); axs[1].set_title("Normal Q-Q", fontsize=9)
    axs[2].bar(range(1, len(r) + 1), m["leverage"], color=ACC)
    axs[2].axhline(2 * m["npar"] / m["n"], c=HOT, ls=":", lw=1)
    axs[2].set_xlabel("Run"); axs[2].set_ylabel("Leverage"); axs[2].set_title("Leverage (dotted: 2p/n)", fontsize=9)
    for a in axs:
        _style(a)
    f.tight_layout()
    return f


def parity(y, m):
    f, ax = _fig(figsize=(3.8, 3.6))
    ax.scatter(y, m["fitted"], c=HOT, s=30)
    lo, hi = min(y.min(), m["fitted"].min()), max(y.max(), m["fitted"].max())
    ax.plot([lo, hi], [lo, hi], c=INK, lw=.8)
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted"); ax.set_title("Parity plot", fontsize=9)
    _style(ax); f.tight_layout()
    return f


def pareto(tab, info):
    f, ax = _fig(figsize=(6.4, 0.38 * len(tab) + 1.4))
    t = tab.iloc[::-1]
    cols = [HOT if v.startswith("significant (B") else ACC if v.startswith("significant") else "#B9A3AD" for v in t["Verdict"]]
    ax.barh(t["Factor"], t["|t|"], color=cols)
    ax.axvline(info["tlim"], c=INK, ls="--", lw=1, label=f"t-limit {info['tlim']:.2f}")
    ax.axvline(info["blim"], c=HOT, ls=":", lw=1.4, label=f"Bonferroni limit {info['blim']:.2f}")
    ax.set_xlabel("|t| of effect"); ax.legend(fontsize=8, loc="lower right", framealpha=.9)
    _style(ax); f.tight_layout()
    return f


def show(st, fig):
    """Render a figure in Streamlit, then close it so a long-running server does not leak memory."""
    st.pyplot(fig)
    plt.close(fig)


def power_curves(sigma, n_runs, df, alpha, marks):
    m = n_runs - 1
    mde_u = rsm.min_detectable_effect(sigma, n_runs, df, alpha, 1)
    mde_b = rsm.min_detectable_effect(sigma, n_runs, df, alpha, m)
    top = mde_b if np.isfinite(mde_b) else mde_u
    xmax = max(2.0, min(12.0, np.ceil(top * 1.7 * 2) / 2))
    x = np.linspace(0, xmax, 81)
    f, ax = _fig(figsize=(7, 3.8))
    ax.plot(x, [rsm.effect_power(v, sigma, n_runs, df, alpha, 1) for v in x], "--", c=ACC, lw=2, label="unadjusted")
    ax.plot(x, [rsm.effect_power(v, sigma, n_runs, df, alpha, m) for v in x], "-", c=HOT, lw=2.2, label="Bonferroni")
    ax.axhline(.8, c=INK, ls=":", lw=1)
    for name, v in marks.items():
        if abs(v) <= xmax:
            ax.axvline(abs(v), c=INK, lw=.8, alpha=.5); ax.text(abs(v) + .02, 1.02, name, fontsize=8, color=INK)
    for v, c in ((mde_u, ACC), (mde_b, HOT)):
        if np.isfinite(v) and v <= xmax:
            ax.scatter([v], [.8], c=c, s=40, zorder=5)
    ax.set_ylim(0, 1.08); ax.set_xlim(0, xmax)
    ax.set_xlabel("True effect size"); ax.set_ylabel("Power"); ax.legend(fontsize=8, loc="lower right")
    _style(ax); f.tight_layout()
    return f, mde_u, mde_b
