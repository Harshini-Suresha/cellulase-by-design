"""Streamlit page: build a run sheet (central composite or Plackett-Burman) in natural units."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

import designs

st.set_page_config(page_title="Design builder", page_icon="🧪", layout="wide")
st.markdown("<style>.stApp{background:#FFE4EC;color:#2A1520}</style>", unsafe_allow_html=True)
st.title("Design builder")
st.caption("Generate a randomised run sheet in real units, ready to take to the bench. Screen with a Plackett–Burman "
           "design, then optimise the important factors with a central composite design.")

kind = st.radio("Design", ["Central composite (response surface)", "Box–Behnken (response surface)", "Two-level full factorial", "Plackett–Burman (screening)"], horizontal=True)

if kind.startswith("Central"):
    k = st.slider("Number of factors", 2, 5, 2)
    c1, c2, c3 = st.columns(3)
    alpha_kind = c1.selectbox("Axial points", ["rotatable", "face-centred", "spherical"],
                              help="Rotatable: equal prediction variance at equal distance from the centre. "
                                   "Face-centred: axial points on the cube faces, so no factor goes beyond its low/high setting.")
    n_c = c2.number_input("Centre-point replicates", 1, 12, 5 if k == 2 else 6)
    half = c3.checkbox("Half-fraction factorial (5 factors only)", k == 5, disabled=k != 5)
    pts, types, a = designs.ccd(k, alpha_kind, int(n_c), 1 if (k == 5 and half) else 0)
elif kind.startswith("Box"):
    k = st.slider("Number of factors", 3, 5, 3)
    n_c = st.number_input("Centre-point replicates", 1, 12, 3)
    pts, types = designs.box_behnken(k, int(n_c)); a = None
    st.caption("Box-Behnken runs never combine all factors at their extremes, so no run is at a corner; there are no axial points beyond the low/high settings.")
elif kind.startswith("Two-level"):
    k = st.slider("Number of factors", 2, 5, 3)
    n_c = st.number_input("Centre-point replicates", 0, 12, 3)
    pts = np.vstack([designs.full_factorial(k), np.zeros((int(n_c), k))])
    types, a = ["Factorial"] * (2 ** k) + ["Centre"] * int(n_c), None
    st.caption("Centre points let you test for curvature and estimate pure error without adding a response-surface design.")
else:
    n_runs = st.selectbox("Runs", [12, 20, 24], help="A design with N runs screens up to N − 1 factors.")
    k = st.slider("Number of real factors", 2, n_runs - 1, min(11, n_runs - 1))
    full = designs.plackett_burman(n_runs)
    pts, types, a = full[:, :k].astype(float), ["Screen"] * n_runs, None
    if k < n_runs - 1:
        st.caption(f"{n_runs - 1 - k} unused column(s) are dropped; they would be dummy factors for estimating error.")

st.subheader("Factor levels")
names, lows, highs = [], [], []
cols = st.columns(min(k, 4))
for j in range(k):
    with cols[j % len(cols)]:
        nm = st.text_input(f"Factor {j+1} name", f"Factor {chr(65+j)}", key=f"n{j}")
        lo = st.number_input("Low (−1)", value=0.0, key=f"l{j}")
        hi = st.number_input("High (+1)", value=1.0, key=f"h{j}")
        names.append(nm); lows.append(lo); highs.append(hi)
if len(set(names)) != k:
    st.error("Give every factor a different name.")
    st.stop()
if any(h <= l for l, h in zip(lows, highs)):
    st.error("Each high level must be greater than its low level.")
    st.stop()

c1, c2 = st.columns(2)
rand = c1.checkbox("Randomise run order", True, help="Randomisation protects against drift and hidden time trends.")
seed = c2.number_input("Random seed", 1, 999999, 2024, disabled=not rand)
df = designs.build(pts, types, names, lows, highs, int(seed) if rand else None)

prop = designs.properties(pts)
m = st.columns(4)
m[0].metric("Runs", prop["runs"]); m[1].metric("Factors", prop["factors"])
m[2].metric("Axial distance α", "n/a" if a is None else f"{a:.3f}")
m[3].metric("Largest off-diagonal of X'X", f"{prop['max_offdiag']:.2g}")
if a is not None and alpha_kind != "face-centred" and a > 1:
    lo_ext = [l - (h - l) / 2 * (a - 1) for l, h in zip(lows, highs)]
    hi_ext = [h + (h - l) / 2 * (a - 1) for l, h in zip(lows, highs)]
    st.info("Axial points reach beyond your low/high settings, to: " +
            "; ".join(f"{n} {lo:.4g} to {hi:.4g}" for n, lo, hi in zip(names, lo_ext, hi_ext)) +
            ". Check these are physically possible, or choose face-centred.")
st.dataframe(df, hide_index=True)
st.download_button("Download run sheet (CSV)", df.to_csv(index=False).encode(), "run_sheet.csv", "text/csv")
st.caption("Fill in the Response column at the bench, then upload the CSV to the analyser pages.")
