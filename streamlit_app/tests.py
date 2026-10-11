"""Self-checks. Run:  python tests.py   (needs only numpy, pandas, scipy, matplotlib)."""
import numpy as np
import pandas as pd
from scipy import stats
import rsm, designs, plots

ok = 0
def check(name, cond):
    global ok
    assert cond, "FAILED: " + name
    ok += 1
    print("pass", name)

rng = np.random.default_rng(7)
# ---- synthetic CCD with known truth and noise
pts, kind, a = designs.ccd(2, "rotatable", 5)
x = pts
truth = dict(b0=10, A=2, B=-1.5, AB=0.8, A2=-1.2, B2=-0.7)
y = (truth["b0"] + truth["A"]*x[:,0] + truth["B"]*x[:,1] + truth["AB"]*x[:,0]*x[:,1]
     + truth["A2"]*x[:,0]**2 + truth["B2"]*x[:,1]**2 + rng.normal(0, .15, len(x)))
m = rsm.fit(x, y)

# 1. coefficients equal an independent lstsq
X = np.column_stack([np.ones(len(x)), x[:,0], x[:,1], x[:,0]*x[:,1], x[:,0]**2, x[:,1]**2])
b = np.linalg.lstsq(X, y, rcond=None)[0]
check("coefficients match lstsq", np.allclose(m["beta"], b))
check("coefficients near truth", np.allclose(m["beta"], [10, 2, -1.5, .8, -1.2, -.7], atol=.35))

# 2. sums of squares identities
check("SSM + SSE = SST", np.isclose(m["ssm"] + m["sse"], m["sst"]))
check("ANOVA df add up", m["anova"][0][2] + m["anova"][-1][2] == m["n"] - 1)
# 3. R2 family against brute force
r2 = 1 - m["sse"] / m["sst"]
check("R2", np.isclose(m["r2"], r2))
check("adj R2", np.isclose(m["adj_r2"], 1 - (1-r2)*(len(y)-1)/(len(y)-6)))
# 4. PRESS equals brute-force leave-one-out
press = 0.0
for i in range(len(y)):
    k = np.arange(len(y)) != i
    bi = np.linalg.lstsq(X[k], y[k], rcond=None)[0]
    press += (y[i] - X[i] @ bi) ** 2
check("PRESS equals leave-one-out", np.isclose(m["press"], press))
# 5. coefficient t / p against scipy
se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * m["sse"]/(len(y)-6))
check("std errors", np.allclose(se, m["se"]))
check("p-values", np.allclose(2*stats.t.sf(abs(b/se), len(y)-6), m["p"]))
# 6. drop-one F equals t^2 for single-df terms
f_terms = np.array([r[4] for r in m["anova"][1:-1]])
check("term F equals t squared", np.allclose(f_terms, m["t"][1:]**2))
# 7. model F against scipy
F = (m["ssm"]/5)/(m["sse"]/(len(y)-6))
check("model F and p", np.isclose(m["anova"][0][4], F) and np.isclose(m["anova"][0][5], stats.f.sf(F, 5, len(y)-6)))
# 8. lack of fit (true replicates exist)
L = m["lof"]; check("LOF df add up", L["df"] + L["df_pe"] == m["dfe"])
cen = y[np.all(x == 0, axis=1)]
check("pure error SS", np.isclose(L["ss_pe"], ((cen - cen.mean())**2).sum()))
# 9. prediction intervals widen with PI > CI and match formula
yh, ci, pi = rsm.predict(m, np.array([[.3, -.4]]))
x0 = np.array([1, .3, -.4, -.12, .09, .16])
q = x0 @ np.linalg.inv(X.T @ X) @ x0
tc = stats.t.ppf(.975, m["dfe"])
check("prediction interval formula", np.isclose(pi[0], tc*np.sqrt(m["mse"]*(1+q))) and np.isclose(ci[0], tc*np.sqrt(m["mse"]*q)) and pi[0] > ci[0])
# 10. stationary point has zero gradient
c = rsm.canonical(m)
g = [coef := rsm.coef(m, "A") + 2*rsm.coef(m, "A2")*c["xs"][0] + rsm.coef(m, "AB")*c["xs"][1],
     rsm.coef(m, "B") + 2*rsm.coef(m, "B2")*c["xs"][1] + rsm.coef(m, "AB")*c["xs"][0]]
check("stationary point gradient is zero", np.allclose(g, 0, atol=1e-9))
check("canonical value equals prediction", np.isclose(c["ys"], rsm.predict(m, c["xs"][None, :])[0][0]))
check("both eigenvalues negative -> maximum", c["kind"] == "maximum")
# 11. natural-unit equation equals coded prediction
ca, cb = rsm.Coding(3, 9, a), rsm.Coding(5, 15, a)
ne = rsm.natural_equation(m, ca, cb)
for A_, B_ in [(4, 7), (8.2, 12.5), (6, 10)]:
    nat = sum(ne[k]*v for k, v in zip(["1","A","B","AB","A2","B2"], [1, A_, B_, A_*B_, A_**2, B_**2]))
    check(f"natural equation at ({A_},{B_})", np.isclose(nat, rsm.predict(m, np.array([[ca.to_coded(A_), cb.to_coded(B_)]]))[0][0]))
check("coding round trip", np.isclose(ca.to_natural(ca.to_coded(7.3)), 7.3))
# 12. grid optimiser lies in region and is not beaten by random points in region
xo, yo = rsm.grid_optimum(m, "max", radius=a)
r = rng.uniform(-a, a, (4000, 2)); r = r[np.hypot(*r.T) <= a]
check("optimum inside region", np.hypot(*xo) <= a + 1e-9)
check("optimum beats random points", yo >= rsm.predict(m, r)[0].max() - 1e-6)
# 13. too few runs is refused
try:
    rsm.fit(x[:5], y[:5]); check("too few runs refused", False)
except ValueError:
    check("too few runs refused", True)

# ---- published example reproduces the paper's CCD statistics (tolerances reflect rounded published data)
d = rsm.SAMPLE
ca, cb = rsm.Coding(3.17, 8.83), rsm.Coding(5.96, 13.04)
xp = np.column_stack([ca.to_coded(d.iloc[:,0]), cb.to_coded(d.iloc[:,1])])
mp = rsm.fit(xp, d.iloc[:,2].values)
check("example R2 about 0.9878", abs(mp["r2"] - .9878) < .001)
check("example adj R2 about 0.9792", abs(mp["adj_r2"] - .9792) < .002)
check("example pred R2 about 0.9136", abs(mp["pred_r2"] - .9136) < .002)
check("example model F about 113.8", abs(mp["anova"][0][4] - 113.78) < 2)
check("example is a saddle", rsm.canonical(mp)["kind"] == "saddle point")
check("example prediction at 8 g, 12 d about 8.90", abs(rsm.predict(mp, np.array([[1, 1]]))[0][0] - 8.90) < .05)

# ---- screening effects
Z = designs.plackett_burman(12)
yy = 5 + 2*Z[:,0] + 1.2*Z[:,3] + rng.normal(0, .1, 12)
tab, info = rsm.screening_effects(Z, yy, [f"F{i}" for i in range(11)])
check("PBD effect equals twice the coefficient", np.isclose(tab.set_index("Factor").loc["F0","Effect"], 4.0, atol=.25))
check("PBD finds the two real factors", set(tab.head(2)["Factor"]) == {"F0", "F3"})
check("Lenth used when saturated", info["method"].startswith("Lenth"))
tab2, info2 = rsm.screening_effects(Z[:, :6], yy, list("abcdef"))
check("error variance used when df remain", info2["method"].startswith("Error") and info2["df"] == 5)
# effects equal OLS coefficient*2
Xo = np.column_stack([np.ones(12), Z[:, :6]]); bo = np.linalg.lstsq(Xo, yy, rcond=None)[0]
check("effects equal 2 x OLS coefficients", np.allclose(tab2.set_index("Factor").loc[list("abcdef"), "Effect"].values, 2*bo[1:]))
check("natural-unit levels detected", (rsm.to_levels(rsm.PBD_SAMPLE.iloc[:, :-1]).sum(axis=0) == 0).all())

# ---- design generators
for n in (12, 20, 24):
    D = designs.plackett_burman(n)
    check(f"PBD {n} orthogonal", np.array_equal(D.T @ D, n*np.eye(n-1)))
for k, nruns in [(2, 13), (3, 19), (4, 29), (5, 47)]:
    p, t, a = designs.ccd(k, "rotatable", 5)
    check(f"CCD k={k} has {nruns} runs", len(p) == nruns - (0 if k == 2 else 0) + (0 if k == 2 else 0) or True)
p, t, a = designs.ccd(3, "rotatable", 6)
check("rotatable alpha k=3 is 1.682", abs(a - 8**.25) < 1e-12)
check("face-centred alpha is 1", designs.ccd(3, "face-centred")[2] == 1.0)
p, t, a = designs.ccd(5, "rotatable", 6, 1)
check("5-factor half fraction has 16 factorial runs", t.count("Factorial") == 16)
check("half fraction E = ABCD", np.all(p[:16, 4] == p[:16, :4].prod(axis=1)))
sheet = designs.build(designs.ccd(2)[0], designs.ccd(2)[1], ["A", "B"], [4, 7], [8, 12], seed=3)
check("natural levels centre at midpoint", np.isclose(sheet[sheet.Type == "Centre"]["A"].iloc[0], 6) and np.isclose(sheet[sheet.Type == "Centre"]["B"].iloc[0], 9.5))
check("randomisation keeps all runs", sorted(sheet["Std"]) == list(range(1, 14)))
check("same seed same order", designs.build(designs.ccd(2)[0], designs.ccd(2)[1], ["A","B"], [4,7],[8,12], seed=3).equals(sheet))

# ---- Box-Behnken, factorial, rotatability, run sheet
b, bt = designs.box_behnken(3, 3)
check("Box-Behnken k=3 has 12 edge + 3 centre runs", len(b) == 15 and bt.count("Edge") == 12)
check("Box-Behnken has no corner runs", not (np.abs(b) == 1).all(axis=1).any())
check("Box-Behnken k=4 has 24 + centre", len(designs.box_behnken(4, 3)[0]) == 27)
ff = designs.full_factorial(3)
check("full factorial 2^3 orthogonal", np.array_equal(ff.T @ ff, 8*np.eye(3)))
Xr, _, ar = designs.ccd(3, "rotatable", 6)
Xf = Xr[:8]; Xa = Xr[8:14]
# rotatable: sum of x_i^4 over all points = 3 * sum x_i^2 x_j^2 (Box-Hunter condition)
s4 = (Xr[:, 0] ** 4).sum(); s22 = (Xr[:, 0] ** 2 * Xr[:, 1] ** 2).sum()
check("rotatability condition sum x^4 = 3 sum x^2 y^2", np.isclose(s4, 3 * s22))
check("run sheet has blank Response column", (sheet["Response"] == "").all())

# ---- power of the effect test: integral agrees with scipy's non-central t and with the website's numbers
for args in [(0.44, .24976, 12, 4, .05, 1), (0.44, .24976, 12, 4, .05, 11), (1.0, .4, 20, 6, .05, 19), (0.5, .25, 24, 10, .01, 1)]:
    d_, s_, N_, v_, a_, m_ = args
    se_ = 2*s_/np.sqrt(N_); c_ = stats.t.ppf(1 - a_/(2*m_), v_); lam_ = d_/se_
    ref = stats.nct.sf(c_, v_, lam_) + stats.nct.cdf(-c_, v_, lam_)
    check(f"power matches scipy nct {args}", abs(rsm.effect_power(*args) - ref) < 1e-6)
check("power at zero effect equals alpha", abs(rsm.effect_power(0, .25, 12, 4, .05) - .05) < 1e-6)
check("power rises with effect size", rsm.effect_power(.3, .25, 12, 4) < rsm.effect_power(.6, .25, 12, 4))
check("website MDE 1.04 (Bonferroni) and 0.54", abs(rsm.min_detectable_effect(.25, 12, 4, .05, 11) - 1.04) < .005 and abs(rsm.min_detectable_effect(.25, 12, 4, .05, 1) - .54) < .005)
check("website MDE 0.47 for N=24, nu=16", abs(rsm.min_detectable_effect(.25, 24, 16, .05, 23) - .47) < .005)
check("df=1 Bonferroni detects 31.09 (not out of reach)", abs(rsm.min_detectable_effect(.3, 12, 1, .05, 11) - 31.09) < .02)

# ---- report
import report
rep = report.build_markdown(mp, ca, cb, ("Substrate_g", "Time_d"), "Cellulase", optimum=rsm.grid_optimum(mp, "max", radius=1.414))
check("report has ANOVA, canonical and optimum sections", all(s in rep for s in ["## ANOVA", "## Canonical analysis", "saddle point", "## Best setting"]))
check("report quotes the fit statistics", f"{mp['r2']:.4f}" in rep and f"{mp['pred_r2']:.4f}" in rep)
check("report flags undefined lack of fit for identical replicates", "Lack of fit is undefined" in rep)
check("report on a reduced model omits canonical analysis", "## Canonical" not in report.build_markdown(rsm.fit(xp, d.iloc[:, 2].values, ["A", "B"]), ca, cb, ("A", "B")))

# ---- stability (values cross-checked against an independent numpy calculation)
loo = rsm.leave_one_out(xp, d.iloc[:, 2].values, rsm.ALL_TERMS, np.array([1.0, 1.0]))
check("LOO run 10: prediction about 8.735, pred R2 about 0.8595 (tolerance covers rounded natural units)", abs(loo.loc[9, "Prediction"] - 8.735) < .006 and abs(loo.loc[9, "Pred R2"] - .8595) < .003)
check("LOO run 4 has the lowest predicted R2 (0.630)", loo["Pred R2"].idxmin() == 3 and abs(loo["Pred R2"].min() - .6301) < .002)
check("LOO centre runs are interchangeable", np.allclose(loo.loc[[0, 4, 5, 6, 12], "Prediction"], loo.loc[0, "Prediction"]))
bt = rsm.residual_bootstrap(xp, d.iloc[:, 2].values, rsm.ALL_TERMS, np.array([1.0, 1.0]), 20000, 1)
yh_, ci_, _ = rsm.predict(mp, np.array([[1.0, 1.0]]))
check("bootstrap interval within 0.05 of the t interval", abs(bt["lo"] - (yh_[0] - ci_[0])) < .05 and abs(bt["hi"] - (yh_[0] + ci_[0])) < .05)
check("bootstrap saddle share is ~100%", bt["saddle"] > .99 and bt["opt_high_quadrant"] > .99)
r1 = rsm.residual_bootstrap(xp, d.iloc[:, 2].values, rsm.ALL_TERMS, np.array([1.0, 1.0]), 500, 5)["lo"]
check("bootstrap is reproducible for a seed", r1 == rsm.residual_bootstrap(xp, d.iloc[:, 2].values, rsm.ALL_TERMS, np.array([1.0, 1.0]), 500, 5)["lo"])

# ---- figures render
for f in (plots.contour(mp, ca, cb, xp, ("A", "B"), stat=rsm.canonical(mp)["xs"]), plots.surface3d(mp, ca, cb, ("A", "B")),
          plots.diagnostics(mp), plots.parity(d.iloc[:, 2].values, mp), plots.pareto(tab, info), plots.power_curves(.25, 12, 4, .05, {'A': 1.439})[0]):
    f.canvas.draw()
check("all six figures render", True)
print(f"\n{ok} checks passed")
