"""Streamlit app: upload a two-factor central composite design and get the full RSM analysis.

Run:  streamlit run app.py
"""
import io
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

import rsm
import plots
import report

st.set_page_config(page_title="RSM analyser", page_icon="📈", layout="wide")
st.markdown("<style>.stApp{background:#FFE4EC;color:#2A1520}</style>", unsafe_allow_html=True)

st.title("Response-surface analyser")
st.caption("Fit a quadratic model to a two-factor central composite design (CCD), read the ANOVA, "
           "locate the optimum and check the assumptions. Runs entirely on the data you give it.")

# ---------------------------------------------------------------- data in
with st.sidebar:
    st.header("1 · Data")
    src = st.radio("Source", ["Published example (cellulase CCD)", "Upload CSV"])
    if src.startswith("Upload"):
        up = st.file_uploader("CSV with three columns: factor A, factor B, response", type="csv")
        df = pd.read_csv(up) if up else None
        st.caption("Natural units, one row per run, header row required. Replicated centre points are encouraged.")
    else:
        df = rsm.SAMPLE.copy()
        st.caption("13-run rotatable CCD, Aspergillus awamori cellulase (Microbial Cell Factories 23:11, 2024).")

if df is None or df.shape[1] < 3:
    st.info("Upload a CSV, or switch to the published example, to begin.")
    st.stop()

cols = list(df.columns)
with st.sidebar:
    ca_, cb_, cy_ = (st.selectbox(l, cols, index=i) for i, l in enumerate(["Factor A column", "Factor B column", "Response column"]))
    alpha = st.number_input("Axial distance α (1 if face-centred, 1.414 if rotatable)", 0.5, 3.0, 1.414, 0.001, format="%.3f")
    st.header("2 · Model")
    terms = ["A", "B"] + [t for t, lab in [("AB", "A×B"), ("A2", "A²"), ("B2", "B²")] if st.checkbox(lab, True)]

d = df[[ca_, cb_, cy_]].apply(pd.to_numeric, errors="coerce").dropna()
if len(d) < 6:
    st.error("Need at least six complete numeric runs.")
    st.stop()
if len({ca_, cb_, cy_}) < 3:
    st.error("Pick three different columns.")
    st.stop()

A, B, Y = (d[c].to_numpy(float) for c in (ca_, cb_, cy_))
CA, CB = rsm.Coding(A.min(), A.max(), alpha), rsm.Coding(B.min(), B.max(), alpha)
X = np.column_stack([CA.to_coded(A), CB.to_coded(B)])
try:
    M = rsm.fit(X, Y, terms)
except Exception as e:  # singular design, too few runs, etc.
    st.error(f"Could not fit the model: {e}")
    st.stop()
names = (ca_, cb_)

# ---------------------------------------------------------------- fit quality
c = st.columns(7)
for col, (lab, val) in zip(c, [("R²", f"{M['r2']:.4f}"), ("Adj. R²", f"{M['adj_r2']:.4f}"), ("Pred. R²", f"{M['pred_r2']:.4f}"),
                               ("Model F", f"{M['anova'][0][4]:.2f}"), ("Model p", "<0.0001" if M['anova'][0][5] < 1e-4 else f"{M['anova'][0][5]:.4f}"),
                               ("CV %", f"{M['cv']:.2f}"), ("Adeq. precision", f"{M['adeq_prec']:.1f}")]):
    col.metric(lab, val)
if M["pred_r2"] < 0 or M["adj_r2"] - M["pred_r2"] > 0.2:
    st.warning("Predicted R² is far below adjusted R²: the model may be over-fitted or an influential run is driving the fit.")

t_model, t_surf, t_opt, t_diag, t_data = st.tabs(["Model & ANOVA", "Surface", "Optimum & prediction", "Diagnostics", "Data & export"])

with t_model:
    an = pd.DataFrame(M["anova"], columns=["Source", "Sum of squares", "df", "Mean square", "F", "p"]).set_index("Source")
    st.subheader("Analysis of variance")
    st.dataframe(an.style.format({"Sum of squares": "{:.4f}", "Mean square": "{:.4f}", "F": "{:.2f}", "p": "{:.4f}"}, na_rep=""))
    L = M["lof"]
    if L and L["ss_pe"] > 0:
        st.write(f"**Lack of fit:** F = {L['F']:.2f} on ({L['df']}, {L['df_pe']}) df, p = {L['p']:.4f}. "
                 + ("Significant lack of fit: the quadratic model is not adequate." if L["p"] < .05 else "Not significant, which is what you want."))
    elif L:
        st.write("**Lack of fit:** the replicates are identical, so pure error is zero and the test is undefined. "
                 "This happens when centre points are reported to the same rounded value.")
    else:
        st.write("**Lack of fit:** not testable (no replicated points).")
    cf = pd.DataFrame({"Coefficient (coded)": M["beta"], "Std. error": M["se"], "t": M["t"], "p": M["p"],
                       "95% CI low": M["beta"] - M["tcrit"] * M["se"], "95% CI high": M["beta"] + M["tcrit"] * M["se"]}, index=M["names"])
    st.subheader("Coefficients")
    st.dataframe(cf.style.format("{:.4f}"))
    ne = rsm.natural_equation(M, CA, CB)
    lab = {"1": "", "A": f"·{ca_}", "B": f"·{cb_}", "AB": f"·{ca_}·{cb_}", "A2": f"·{ca_}²", "B2": f"·{cb_}²"}
    st.subheader("Equation in natural units")
    st.code("y = " + " ".join(f"{v:+.5g}{lab[k]}" for k, v in ne.items() if k == "1" or k in M["terms"]), language="text")
    st.caption(f"Coding: A = ({ca_} − {CA.centre:.4g}) / {CA.step:.4g},  B = ({cb_} − {CB.centre:.4g}) / {CB.step:.4g}.")

can = rsm.canonical(M) if {"A2", "B2", "AB"} <= set(M["terms"]) else None
with t_surf:
    l, r = st.columns(2)
    with l:
        plots.show(st, plots.contour(M, CA, CB, X, names, stat=None if can is None else can["xs"]))
    with r:
        e, z = st.columns(2)
        plots.show(st, plots.surface3d(M, CA, CB, names, elev=e.slider("Elevation", 5, 80, 26), azim=z.slider("Azimuth", -180, 180, -58)))
    if can:
        st.subheader("Canonical analysis")
        inside = np.abs(can["xs"]).max() <= alpha + 1e-9
        st.write(f"Stationary point: **{CA.to_natural(can['xs'][0]):.3f}**, **{CB.to_natural(can['xs'][1]):.3f}** "
                 f"({ca_}, {cb_}), predicted {can['ys']:.3f}. Eigenvalues {can['eig'][0]:.3f} and {can['eig'][1]:.3f}: a **{can['kind']}**"
                 + ("" if inside else ", outside the design region (extrapolation)") + ". "
                 + ("A saddle means the best response is on the edge of the region, not at the stationary point." if can["kind"] == "saddle point" else ""))

with t_opt:
    st.subheader("Predict at a chosen setting")
    p1, p2 = st.columns(2)
    va = p1.number_input(ca_, float(A.min()), float(A.max()), float(CA.centre))
    vb = p2.number_input(cb_, float(B.min()), float(B.max()), float(CB.centre))
    pt = np.array([CA.to_coded(va), CB.to_coded(vb)])
    yh, ci, pi = rsm.predict(M, pt[None, :])
    inside = np.hypot(*pt) <= float(np.max(np.hypot(X[:, 0], X[:, 1]))) + 1e-9
    st.metric("Predicted response", f"{yh[0]:.3f}", f"95% PI {yh[0]-pi[0]:.3f} to {yh[0]+pi[0]:.3f}", delta_color="off")
    st.caption(f"95% confidence interval for the mean: {yh[0]-ci[0]:.3f} to {yh[0]+ci[0]:.3f}. "
               + ("Inside the design region." if inside else "Outside the design region: this is extrapolation."))
    st.divider()
    st.subheader("Find the best setting")
    mode = st.radio("Goal", ["Maximise", "Minimise", "Hit a target"], horizontal=True)
    tgt = st.number_input("Target", value=float(np.median(Y))) if mode == "Hit a target" else None
    restrict = st.checkbox("Stay inside the design region (circle through the outermost points)", True)
    rad = float(np.max(np.hypot(X[:, 0], X[:, 1]))) if restrict else None
    xo, yo = rsm.grid_optimum(M, {"Maximise": "max", "Minimise": "min", "Hit a target": "target"}[mode], tgt, rad, lim=alpha * 1.2)
    _, cio, pio = rsm.predict(M, xo[None, :])
    st.success(f"{ca_} = {CA.to_natural(xo[0]):.3f}, {cb_} = {CB.to_natural(xo[1]):.3f} → predicted {yo:.3f} "
               f"(95% PI {yo-pio[0]:.3f} to {yo+pio[0]:.3f}). Confirm with replicate runs before relying on it.")

with t_diag:
    plots.show(st, plots.diagnostics(M))
    out = np.abs(M["stud"]) > 3
    st.write("Runs with |studentised residual| > 3: " + (", ".join(str(i + 1) for i in np.where(out)[0]) if out.any() else "none") + ".")
    left, _ = st.columns([1, 2])
    plots.show(left, plots.parity(Y, M))
    st.subheader("Stability: leave-one-out and bootstrap")
    pt_c = np.array([1.0, 1.0]) if {"A2", "B2", "AB"} <= set(M["terms"]) else np.array([0.0, 0.0])
    loo = rsm.leave_one_out(X, Y, M["terms"], pt_c)
    boot = rsm.residual_bootstrap(X, Y, M["terms"], pt_c, 2000, 2024)
    yh0, ci0, _ = rsm.predict(M, pt_c[None, :])
    b1, b2, b3 = st.columns(3)
    b1.metric("Prediction at the (+1, +1) corner" if pt_c[0] else "Prediction at the centre", f"{yh0[0]:.3f}")
    b2.metric("Bootstrap 95% interval", f"{boot['lo']:.2f} to {boot['hi']:.2f}")
    b3.metric("t-based 95% CI", f"{yh0[0]-ci0[0]:.2f} to {yh0[0]+ci0[0]:.2f}")
    if "saddle" in boot:
        st.write(f"Saddle in {100*boot['saddle']:.1f}% of 2000 refits; optimum with A>0 and B>0 in {100*boot['opt_high_quadrant']:.1f}%.")
    st.dataframe(loo.style.format({"R2": "{:.4f}", "Pred R2": "{:.4f}", "Prediction": "{:.3f}"}), hide_index=True)
    st.caption("Each row refits the model without that run. A run whose removal sharply lowers predicted R² is influential.")

with t_data:
    res = d.copy()
    res["Predicted"], res["Residual"], res["Studentised"], res["Leverage"] = M["fitted"], M["resid"], M["stud"], M["leverage"]
    st.dataframe(res.style.format("{:.4f}"))
    st.download_button("Download runs and residuals (CSV)", res.to_csv(index=False).encode(), "rsm_results.csv", "text/csv")
    st.download_button("Download ANOVA (CSV)", an.to_csv().encode(), "rsm_anova.csv", "text/csv")
    xo_r, yo_r = rsm.grid_optimum(M, "max", None, float(np.max(np.hypot(X[:, 0], X[:, 1]))), lim=alpha * 1.2)
    st.download_button("Download analysis report (Markdown)", report.build_markdown(M, CA, CB, names, cy_, optimum=(xo_r, yo_r)).encode(), "rsm_report.md", "text/markdown")
    st.download_button("Download the example data (CSV)", rsm.SAMPLE.to_csv(index=False).encode(), "example_ccd.csv", "text/csv")
