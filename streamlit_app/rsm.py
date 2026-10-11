"""Two-factor response-surface (quadratic) analysis. Only numpy, pandas, scipy."""
import numpy as np
import pandas as pd
from scipy import stats

ALL_TERMS = ["A", "B", "AB", "A2", "B2"]


def design_matrix(x, terms):
    a, b = x[:, 0], x[:, 1]
    cols = {"1": np.ones(len(a)), "A": a, "B": b, "AB": a * b, "A2": a * a, "B2": b * b}
    names = ["1"] + [t for t in ALL_TERMS if t in terms]
    return np.column_stack([cols[n] for n in names]), names


def fit(x, y, terms=ALL_TERMS):
    """OLS fit in coded units. Returns a dict of every statistic the app shows."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X, names = design_matrix(x, terms)
    n, p = X.shape
    if n <= p:
        raise ValueError(f"{n} runs cannot support a model with {p} parameters.")
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    fitted = X @ beta
    resid = y - fitted
    dfe = n - p
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    ssm = sst - sse
    mse = sse / dfe
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    press = float(((resid / (1 - h)) ** 2).sum())
    se = np.sqrt(np.diag(XtXi) * mse)
    tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), dfe)
    tcrit = stats.t.ppf(0.975, dfe)

    # Term significance: drop-one (type III) extra sum of squares
    rows = []
    for t in names[1:]:
        keep = [u for u in names[1:] if u != t]
        Xr, _ = design_matrix(x, keep)
        rr = y - Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
        ss = float(rr @ rr) - sse
        f = ss / mse
        rows.append((t, ss, 1, ss / 1, f, stats.f.sf(f, 1, dfe)))
    fm = (ssm / (p - 1)) / mse
    anova = [("Model", ssm, p - 1, ssm / (p - 1), fm, stats.f.sf(fm, p - 1, dfe))] + rows
    anova.append(("Residual", sse, dfe, mse, np.nan, np.nan))

    # Lack of fit from replicated design points
    lof = None
    keys = pd.Series([tuple(np.round(r, 6)) for r in x])
    grp = pd.Series(y).groupby(keys.values)
    sspe = float(sum(((g - g.mean()) ** 2).sum() for _, g in grp))
    dfpe = int(sum(len(g) - 1 for _, g in grp))
    if dfpe > 0 and dfe - dfpe > 0:
        sslof, dflof = sse - sspe, dfe - dfpe
        if sspe > 0:
            f = (sslof / dflof) / (sspe / dfpe)
            lof = dict(ss=sslof, df=dflof, ss_pe=sspe, df_pe=dfpe, F=f, p=float(stats.f.sf(f, dflof, dfpe)))
        else:
            lof = dict(ss=sslof, df=dflof, ss_pe=0.0, df_pe=dfpe, F=np.nan, p=np.nan)
    r2 = ssm / sst
    pred = np.array(fitted) + 0
    return dict(
        names=names, beta=beta, se=se, t=tval, p=pval, tcrit=tcrit, dfe=dfe, n=n, npar=p,
        mse=mse, sse=sse, sst=sst, ssm=ssm, anova=anova, lof=lof,
        r2=r2, adj_r2=1 - (1 - r2) * (n - 1) / dfe, pred_r2=1 - press / sst, press=press,
        cv=100 * np.sqrt(mse) / y.mean(), fitted=fitted, resid=resid, leverage=h,
        stud=resid / np.sqrt(mse * (1 - h)), XtXi=XtXi, terms=list(terms),
        adeq_prec=(pred.max() - pred.min()) / np.sqrt(p * mse / n),
    )


def predict(m, pts, level=0.95):
    """Mean prediction with confidence and prediction intervals at coded points (k x 2)."""
    X, _ = design_matrix(np.atleast_2d(pts), m["terms"])
    yhat = X @ m["beta"]
    q = np.einsum("ij,jk,ik->i", X, m["XtXi"], X)
    tc = stats.t.ppf(0.5 + level / 2, m["dfe"])
    ci = tc * np.sqrt(m["mse"] * q)
    pi = tc * np.sqrt(m["mse"] * (1 + q))
    return yhat, ci, pi


def coef(m, name):
    return float(m["beta"][m["names"].index(name)]) if name in m["names"] else 0.0


def canonical(m):
    """Stationary point and eigen-analysis of the fitted quadratic (coded units)."""
    b = np.array([coef(m, "A"), coef(m, "B")])
    B = np.array([[coef(m, "A2"), coef(m, "AB") / 2], [coef(m, "AB") / 2, coef(m, "B2")]])
    if abs(np.linalg.det(B)) < 1e-12:
        return None
    xs = -0.5 * np.linalg.solve(B, b)
    ys = float(coef(m, "1") + 0.5 * b @ xs)
    ev = np.linalg.eigvalsh(B)
    kind = ("maximum" if (ev < 0).all() else "minimum" if (ev > 0).all() else "saddle point")
    return dict(xs=xs, ys=ys, eig=ev, kind=kind)


def grid_optimum(m, mode="max", target=None, radius=None, lim=1.414, n=161):
    """Best point on a grid; optionally restricted to a circle (rotatable design region)."""
    g = np.linspace(-lim, lim, n)
    A, B = np.meshgrid(g, g)
    P = np.column_stack([A.ravel(), B.ravel()])
    if radius is not None:
        P = P[np.hypot(P[:, 0], P[:, 1]) <= radius + 1e-9]
    yh = predict(m, P)[0]
    i = {"max": np.argmax(yh), "min": np.argmin(yh), "target": np.argmin(np.abs(yh - (target or 0)))}[mode]
    return P[i], float(yh[i])


class Coding:
    """Maps natural units <-> coded units: coded = (natural - centre) / step."""
    def __init__(self, lo, hi, alpha=1.414):
        self.centre = (lo + hi) / 2
        self.step = (hi - lo) / 2 / alpha

    def to_coded(self, v):
        return (np.asarray(v, float) - self.centre) / self.step

    def to_natural(self, c):
        return self.centre + self.step * np.asarray(c, float)


def natural_equation(m, ca, cb):
    """Coefficients in natural units for readers who want the equation to use at the bench."""
    # y = b0 + bA a + bB b + bAB ab + bAA a2 + bBB b2, with a=(A-ca.centre)/ca.step
    sa, sb, ma, mb = ca.step, cb.step, ca.centre, cb.centre
    b0, bA, bB, bAB, bAA, bBB = (coef(m, k) for k in ["1", "A", "B", "AB", "A2", "B2"])
    return {
        "1": b0 - bA * ma / sa - bB * mb / sb + bAB * ma * mb / (sa * sb) + bAA * ma**2 / sa**2 + bBB * mb**2 / sb**2,
        "A": bA / sa - bAB * mb / (sa * sb) - 2 * bAA * ma / sa**2,
        "B": bB / sb - bAB * ma / (sa * sb) - 2 * bBB * mb / sb**2,
        "AB": bAB / (sa * sb), "A2": bAA / sa**2, "B2": bBB / sb**2,
    }


SAMPLE = pd.DataFrame(
    [(6, 9.5, 6.85), (3.17, 9.5, 6.15), (8.83, 9.5, 8.12), (4, 12, 4.49), (6, 9.5, 6.85), (6, 9.5, 6.85),
     (6, 9.5, 6.85), (4, 7, 7.85), (6, 13.04, 6.67), (8, 12, 9.00), (8, 7, 7.00), (6, 5.96, 8.00), (6, 9.5, 6.85)],
    columns=["Substrate_g", "Time_d", "Cellulase_U_per_mL"],
)


# ----------------------------------------------------------------------------- screening designs
def screening_effects(Z, y, names):
    """Main effects of a two-level screening design (Plackett-Burman, fractional factorial).

    Z is the n x k matrix of coded levels (-1/+1). Effect = mean(high) - mean(low).
    Significance: if the design leaves residual degrees of freedom (n - 1 - k > 0) the effects are
    tested with the ordinary error variance; if it is saturated (e.g. 12 runs, 11 factors) Lenth's
    pseudo standard error is used instead. Also returns the Bonferroni-adjusted limit.
    """
    Z, y = np.asarray(Z, float), np.asarray(y, float)
    n, k = Z.shape
    eff = np.array([y[Z[:, j] > 0].mean() - y[Z[:, j] < 0].mean() for j in range(k)])
    dfe = n - 1 - k
    if dfe > 0:
        X = np.column_stack([np.ones(n), Z])
        r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        se = np.sqrt((r @ r) / dfe * 4 / n)  # SE of an effect (twice the coefficient SE)
        method, df = "Error variance", dfe
    else:
        a = np.abs(eff)
        s0 = 1.5 * np.median(a)
        pse = 1.5 * np.median(a[a < 2.5 * s0]) if (a < 2.5 * s0).any() else s0
        se, method, df = pse, "Lenth pseudo standard error", k / 3
    t = eff / se
    tlim = stats.t.ppf(0.975, df)
    blim = stats.t.ppf(1 - 0.05 / (2 * k), df)
    p = 2 * stats.t.sf(np.abs(t), df)
    out = pd.DataFrame({"Factor": names, "Effect": eff, "|t|": np.abs(t), "p": p})
    out["Verdict"] = np.where(out["|t|"] > blim, "significant (Bonferroni)", np.where(out["|t|"] > tlim, "significant (t-limit)", "not significant"))
    return out.sort_values("|t|", ascending=False).reset_index(drop=True), dict(method=method, df=df, se=se, tlim=tlim, blim=blim)


def to_levels(df_factors):
    """Convert natural-unit factor columns to -1/+1 by comparing each value with the column midpoint."""
    Z = np.zeros(df_factors.shape)
    for j, c in enumerate(df_factors.columns):
        v = df_factors[c].to_numpy(float)
        Z[:, j] = np.where(v > (v.min() + v.max()) / 2, 1.0, -1.0)
    return Z


PBD_SAMPLE = pd.DataFrame(
    [[2,4,10,0,1,1,0,.01,.01,.01,0,0.929],[2,4,0,0,0,0,0,0,0,0,0,1.864],[2,4,0,10,0,1,1,0,.01,.01,.01,2.053],
     [4,7,0,10,1,1,0,0,0,.01,0,3.727],[4,7,0,0,0,1,0,.01,.01,0,.01,4.013],[2,7,0,10,1,0,1,.01,.01,0,0,2.522],
     [4,4,10,10,1,0,0,0,.01,0,.01,2.843],[4,4,0,0,1,0,1,.01,0,.01,.01,3.507],[4,4,10,10,0,1,1,.01,0,0,0,2.613],
     [2,7,10,0,1,1,1,0,0,0,.01,2.369],[2,7,10,10,0,0,0,.01,0,.01,.01,2.969],[4,7,10,0,0,0,1,0,.01,.01,0,4.636]],
    columns=["Substrate weight", "Time", "Glucose", "Lactose", "Baker's yeast", "Peptone", "(NH4)2SO4", "CuSO4", "FeSO4", "CaCl2", "KCl", "Cellulase_U_per_mL"],
)


# ----------------------------------------------------------------------------- power of a screening effect test
def effect_power(delta, sigma, n_runs, df, alpha=0.05, m=1):
    """Power of the two-sided t-test for a two-level-design effect of true size delta.

    SE(effect) = 2*sigma/sqrt(n_runs); critical value uses alpha/m (Bonferroni over m tests).
    Computed as P(|T'|>c) with T' = (Z+lambda)/sqrt(V/df), V ~ chi-square(df), by numerical
    integration over V (scipy's non-central t returns NaN for large non-centrality).
    """
    from scipy import integrate
    se = 2 * sigma / np.sqrt(n_runs)
    c = stats.t.ppf(1 - alpha / (2 * m), df)
    lam = abs(delta) / se

    def f(v):
        r = c * np.sqrt(v / df)
        return stats.chi2.pdf(v, df) * (stats.norm.cdf(lam - r) + stats.norm.cdf(-lam - r))

    return float(integrate.quad(f, 0, np.inf, limit=500)[0])


def min_detectable_effect(sigma, n_runs, df, alpha=0.05, m=1, power=0.8, hi=100.0):
    """Smallest true effect detected with the requested power; inf if beyond hi."""
    from scipy import optimize
    if effect_power(hi, sigma, n_runs, df, alpha, m) < power:
        return float("inf")
    return float(optimize.brentq(lambda d: effect_power(d, sigma, n_runs, df, alpha, m) - power, 1e-6, hi))


# ----------------------------------------------------------------------------- stability: leave-one-out and bootstrap
def leave_one_out(x, y, terms, point):
    """Refit with each run removed; return a DataFrame of R2, predicted R2 and the prediction at `point` (coded)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rows = []
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        m = fit(x[k], y[k], terms)
        rows.append((i + 1, m["r2"], m["pred_r2"], float(predict(m, np.atleast_2d(point))[0][0])))
    return pd.DataFrame(rows, columns=["Run removed", "R2", "Pred R2", "Prediction"])


def residual_bootstrap(x, y, terms, point, n_boot=2000, seed=2024):
    """Residual bootstrap of the prediction at `point`. Residuals are rescaled by 1/sqrt(1-h) and centred.

    Returns dict with the percentile interval, SD, the share of refits with a saddle stationary point and,
    when the full quadratic is fitted, the share whose best grid point in the design circle has A>0 and B>0.
    """
    m = fit(x, y, terms)
    X, _ = design_matrix(np.asarray(x, float), terms)
    rs = m["resid"] / np.sqrt(1 - m["leverage"]); rs = rs - rs.mean()
    rng = np.random.default_rng(seed)
    n = len(y)
    Y = m["fitted"][None, :] + rs[rng.integers(0, n, (n_boot, n))]
    BB = (m["XtXi"] @ X.T @ Y.T).T
    x0, _ = design_matrix(np.atleast_2d(point), terms)
    P = BB @ x0[0]
    out = dict(lo=float(np.percentile(P, 2.5)), hi=float(np.percentile(P, 97.5)), sd=float(P.std(ddof=1)), samples=P)
    if {"A2", "B2", "AB"} <= set(terms):
        c = {t: BB[:, m["names"].index(t)] for t in m["names"]}
        det_tr = (c["A2"] + c["B2"]) / 2
        d = np.sqrt(((c["A2"] - c["B2"]) / 2) ** 2 + c["AB"] ** 2 / 4)
        out["saddle"] = float(((det_tr + d > 0) & (det_tr - d < 0)).mean())
        g = np.linspace(-1.414, 1.414, 21); A, B = np.meshgrid(g, g)
        msk = np.hypot(A, B) <= 1.4143; A, B = A[msk], B[msk]
        Z = c["1"][:, None] + c["A"][:, None] * A + c["B"][:, None] * B + c["AB"][:, None] * A * B + c["A2"][:, None] * A * A + c["B2"][:, None] * B * B
        i = Z.argmax(1)
        out["opt_high_quadrant"] = float(((A[i] > 0) & (B[i] > 0)).mean())
    return out
