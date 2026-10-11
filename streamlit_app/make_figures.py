"""Publication-style figures (white ground, colour-blind-safe palette, 300 dpi PNG + vector SVG/PDF).

Run:  python make_figures.py [outdir]
Every figure is recomputed from the data in rsm.py, except Figure 1, whose bar heights were digitised
from the published chart (Microbial Cell Factories 23:11) and are marked as such.
"""
import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import rsm

# Okabe-Ito palette
BLUE, ORANGE, GREEN, VERM, SKY, PURPLE, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#7F7F7F"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "axes.linewidth": .8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": .8, "ytick.major.width": .8,
    "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False, "legend.fontsize": 7.5,
    "figure.dpi": 100, "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none",
})
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
OUT.mkdir(exist_ok=True, parents=True)
CA, CB = rsm.Coding(3.17, 8.83), rsm.Coding(5.96, 13.04)


def tag(ax, s, dx=-.14, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=.12, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def fig1_screening():
    lab = ["RS", "OP", "LP", "CC", "MS", "PP", "AL"]
    mean = [1.06, .48, .75, .88, 2.04, 1.62, 1.75]
    sd = [.31, .06, .06, .07, .26, .28, .22]
    f, ax = plt.subplots(figsize=(3.5, 2.6))
    cols = [GREY] * 7; cols[4] = BLUE
    ax.bar(lab, mean, .62, yerr=sd, color=cols, edgecolor="black", linewidth=.6, capsize=2.5, error_kw=dict(lw=.8))
    ax.set_ylabel("Cellulase activity (U mL$^{-1}$)"); ax.set_xlabel("Agricultural waste")
    ax.set_ylim(0, 2.6); ax.yaxis.set_major_locator(MultipleLocator(.5))
    ax.text(.99, .98, "values digitised from the\npublished chart; mean ± SD", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=GREY)
    save(f, "fig1_substrate_screening")


def fig2_pareto():
    Z, y, names = _pbd()
    keep = [0, 1, 4, 5, 6, 9, 10]  # the 7 factors analysed in the study (error df = 4)
    tab, info = rsm.screening_effects(Z[:, keep], y, [names[i] for i in keep])
    from scipy import stats
    info["blim"] = stats.t.ppf(1 - 0.05 / (2 * 11), info["df"])  # Bonferroni over all 11 screened factors, as in the study
    f, ax = plt.subplots(figsize=(3.5, 3.0))
    d = tab.iloc[::-1]
    cols = [VERM if v > info["blim"] else BLUE if v > info["tlim"] else "#BBBBBB" for v in d["|t|"]]
    ax.barh(d["Factor"], d["|t|"], color=cols, edgecolor="black", linewidth=.4, height=.7)
    ax.axvline(info["tlim"], c="black", ls="--", lw=.8)
    ax.axvline(info["blim"], c=VERM, ls=":", lw=1.1)
    ax.text(info["tlim"], len(d) - .3, f" t = {info['tlim']:.2f}", fontsize=6.5, va="bottom")
    ax.text(info["blim"], len(d) - .3, f" Bonferroni = {info['blim']:.2f}", fontsize=6.5, va="bottom", color=VERM)
    ax.set_xlabel("|t| of effect"); ax.set_ylim(-.6, len(d) + .6)
    save(f, "fig2_pareto")


def _pbd():
    d = rsm.PBD_SAMPLE
    names = list(d.columns[:-1])
    return rsm.to_levels(d[names]), d.iloc[:, -1].to_numpy(float), names


def _ccd():
    d = rsm.SAMPLE
    x = np.column_stack([CA.to_coded(d.iloc[:, 0]), CB.to_coded(d.iloc[:, 1])])
    y = d.iloc[:, 2].to_numpy(float)
    return x, y, rsm.fit(x, y)


def fig3_contour_parity():
    x, y, m = _ccd()
    g = np.linspace(-1.414, 1.414, 161)
    A, B = np.meshgrid(g, g)
    Z = rsm.predict(m, np.column_stack([A.ravel(), B.ravel()]))[0].reshape(A.shape)
    f, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=.38))
    cf = a.contourf(CA.to_natural(A), CB.to_natural(B), Z, 14, cmap="viridis")
    cs = a.contour(CA.to_natural(A), CB.to_natural(B), Z, 7, colors="white", linewidths=.5)
    a.clabel(cs, fontsize=6, fmt="%.1f")
    t = np.linspace(0, 2 * np.pi, 200)
    a.plot(CA.to_natural(1.414 * np.cos(t)), CB.to_natural(1.414 * np.sin(t)), c="white", ls="--", lw=.7)
    a.scatter(CA.to_natural(x[:, 0]), CB.to_natural(x[:, 1]), s=14, c="white", edgecolor="black", linewidth=.6, zorder=5)
    a.set_xlabel("Substrate weight (g)"); a.set_ylabel("Incubation time (d)")
    cb = f.colorbar(cf, ax=a, pad=.02); cb.set_label("Cellulase (U mL$^{-1}$)", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    tag(a, "A", -.2)
    b.scatter(y, m["fitted"], s=16, c=BLUE, edgecolor="black", linewidth=.4, zorder=3)
    lo, hi = min(y.min(), m["fitted"].min()) - .2, max(y.max(), m["fitted"].max()) + .2
    b.plot([lo, hi], [lo, hi], c="black", lw=.7); b.set_xlim(lo, hi); b.set_ylim(lo, hi); b.set_aspect("equal")
    b.set_xlabel("Observed (U mL$^{-1}$)"); b.set_ylabel("Predicted (U mL$^{-1}$)")
    b.text(.05, .93, f"$R^2$ = {m['r2']:.3f}\nadj. $R^2$ = {m['adj_r2']:.3f}\npred. $R^2$ = {m['pred_r2']:.3f}", transform=b.transAxes, va="top", fontsize=7)
    tag(b, "B", -.28)
    save(f, "fig3_contour_parity")


def fig4_surface():
    x, y, m = _ccd()
    g = np.linspace(-1.414, 1.414, 71)
    A, B = np.meshgrid(g, g)
    Z = rsm.predict(m, np.column_stack([A.ravel(), B.ravel()]))[0].reshape(A.shape)
    f = plt.figure(figsize=(3.6, 3.1))
    ax = f.add_subplot(projection="3d")
    ax.plot_surface(CA.to_natural(A), CB.to_natural(B), Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.scatter(CA.to_natural(x[:, 0]), CB.to_natural(x[:, 1]), y, c="white", edgecolor="black", s=14, depthshade=False)
    ax.view_init(24, -58)
    ax.set_xlabel("Substrate (g)", fontsize=7, labelpad=-2); ax.set_ylabel("Time (d)", fontsize=7, labelpad=-2)
    ax.set_zlabel("U mL$^{-1}$", fontsize=7, labelpad=-4); ax.tick_params(labelsize=6, pad=-2)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor("white"); a.pane.set_edgecolor("#DDDDDD")
    save(f, "fig4_response_surface")


def fig5_diagnostics():
    from scipy import stats
    x, y, m = _ccd()
    r = m["stud"]
    f, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=.45))
    ax[0].scatter(m["fitted"], r, s=14, c=BLUE, edgecolor="black", linewidth=.4)
    ax[0].axhline(0, c="black", lw=.6)
    for v in (-2, 2): ax[0].axhline(v, c=VERM, ls=":", lw=.9)
    ax[0].set_xlabel("Fitted (U mL$^{-1}$)"); ax[0].set_ylabel("Studentised residual")
    q = stats.norm.ppf((np.arange(1, len(r) + 1) - .5) / len(r))
    ax[1].scatter(q, np.sort(r), s=14, c=BLUE, edgecolor="black", linewidth=.4)
    lim = max(abs(q).max(), abs(r).max()) * 1.05
    ax[1].plot([-lim, lim], [-lim, lim], c=VERM, lw=.9)
    ax[1].set_xlabel("Normal quantile"); ax[1].set_ylabel("Ordered residual")
    ax[2].bar(range(1, len(r) + 1), m["leverage"], color=BLUE, edgecolor="black", linewidth=.4)
    ax[2].axhline(2 * m["npar"] / m["n"], c=VERM, ls=":", lw=.9)
    ax[2].set_xlabel("Run"); ax[2].set_ylabel("Leverage")
    for a, s in zip(ax, "ABC"): tag(a, s, -.33)
    save(f, "fig5_diagnostics")


def fig6_interaction():
    x, y, m = _ccd()
    f, ax = plt.subplots(figsize=(3.5, 2.6))
    tt = np.linspace(-1.414, 1.414, 100)
    for a, col, ls in ((-1, BLUE, "-"), (0, GREY, "--"), (1, VERM, "-")):
        z = rsm.predict(m, np.column_stack([np.full_like(tt, a), tt]))[0]
        ax.plot(CB.to_natural(tt), z, c=col, ls=ls, lw=1.4, label=f"{CA.to_natural(a):.1f} g")
    ax.set_xlabel("Incubation time (d)"); ax.set_ylabel("Predicted cellulase (U mL$^{-1}$)")
    ax.legend(title="Substrate", title_fontsize=7, loc="upper right")
    save(f, "fig6_interaction")


def fig7_power():
    Z, y, names = _pbd()
    keep = [0, 1, 4, 5, 6, 9, 10]
    X = np.column_stack([np.ones(12), Z[:, keep]])
    r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    sigma = float(np.sqrt(r @ r / (12 - 8)))
    xs = np.linspace(0, 3.5, 71)
    f, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(xs, [rsm.effect_power(v, sigma, 12, 4, .05, 1) for v in xs], ls="--", c=BLUE, lw=1.4, label="unadjusted")
    ax.plot(xs, [rsm.effect_power(v, sigma, 12, 4, .05, 11) for v in xs], c=VERM, lw=1.6, label="Bonferroni")
    ax.axhline(.8, c="black", ls=":", lw=.8)
    ax.set_xlabel("True effect (U mL$^{-1}$)"); ax.set_ylabel("Power"); ax.set_ylim(0, 1.02); ax.legend(loc="lower right")
    save(f, "fig7_power")


def fig8_sem():
    import cv2
    src = cv2.imread(str(pathlib.Path(__file__).parent / "assets" / "sem_source.png"))
    if src is None:
        print("sem_source.png missing, skipped"); return
    # panel boxes in the 713x522 source (excluding labels' white boxes and selection handles)
    boxes = {"A": (22, 22, 362, 270), "B": (378, 22, 684, 270), "C": (22, 283, 362, 492), "D": (380, 290, 676, 503)}
    # original in-image letter boxes (x0, y0, x1, y1): covered and replaced by a uniform label
    lab = {"A": (50, 30, 94, 79), "B": (398, 50, 444, 96), "C": (70, 304, 112, 341), "D": (400, 296, 442, 340)}
    # erase green dot and its connector in B
    mask = np.zeros(src.shape[:2], np.uint8)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask[(hsv[..., 0] > 40) & (hsv[..., 0] < 85) & (hsv[..., 1] > 120)] = 255
    mask[255:285, 522:535] = 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    src = cv2.inpaint(src, mask, 4, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    f, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), gridspec_kw=dict(wspace=.03, hspace=.05))
    for ax, k in zip(axs.ravel(), "ABCD"):
        x0, y0, x1, y1 = boxes[k]
        ax.imshow(rgb[y0:y1, x0:x1]); ax.axis("off")
        lx0, ly0, lx1, ly1 = lab[k]
        ax.add_patch(plt.Rectangle((lx0 - x0, ly0 - y0), lx1 - lx0, ly1 - ly0, fc="white", ec="none", zorder=3))
        ax.text((lx0 + lx1) / 2 - x0, (ly0 + ly1) / 2 - y0, k, fontsize=13, fontweight="bold", ha="center", va="center", zorder=4)
    save(f, "fig8_sem_plate")


if __name__ == "__main__":
    for fn in (fig1_screening, fig2_pareto, fig3_contour_parity, fig4_surface, fig5_diagnostics, fig6_interaction, fig7_power, fig8_sem):
        fn()
