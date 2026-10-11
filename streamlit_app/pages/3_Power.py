"""Streamlit page: power and minimum detectable effect for a two-level screening design."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

import rsm
import plots

st.set_page_config(page_title="Power explorer", page_icon="⚡", layout="wide")
st.markdown("<style>.stApp{background:#FFE4EC;color:#2A1520}</style>", unsafe_allow_html=True)
st.title("Power and minimum detectable effect")
st.caption("How large must a factor's effect be before a screening design reliably finds it? "
           "'Not significant' only means something if the design could have seen an effect of that size.")

with st.sidebar:
    n = st.selectbox("Runs, N", [12, 20, 24])
    sigma = st.number_input("Noise σ (response units)", 0.01, 50.0, 0.25, 0.005, format="%.3f",
                            help="Residual standard deviation of one run. Take it from replicates or a pilot fit.")
    df = st.slider("Error degrees of freedom, ν", 1, 30, 4)
    alpha = st.selectbox("Significance level α", [0.05, 0.01, 0.10])
    st.caption("Defaults reproduce the cellulase screen: N = 12, ν = 4, σ ≈ 0.25 U/mL.")

obs = st.text_input("Optional: estimated effects to mark (name=value, comma separated)", "A=1.439, B=1.071, F=-0.44, E=-0.375")
marks = {}
for part in obs.split(","):
    if "=" in part:
        k, v = part.split("=", 1)
        try:
            marks[k.strip()] = float(v)
        except ValueError:
            st.warning(f"Could not read '{part.strip()}'.")

fig, mde_u, mde_b = plots.power_curves(float(sigma), int(n), int(df), float(alpha), marks)
c1, c2, c3 = st.columns(3)
c1.metric("SE of an effect", f"{2 * sigma / np.sqrt(n):.3f}")
c2.metric("Detectable effect (80% power), unadjusted", f"{mde_u:.2f}" if np.isfinite(mde_u) else "out of reach")
c3.metric(f"Detectable effect, Bonferroni (m = {n - 1})", f"{mde_b:.2f}" if np.isfinite(mde_b) else "out of reach")
plots.show(st, fig)
if marks:
    rows = [(k, v, 100 * rsm.effect_power(v, sigma, n, df, alpha, 1), 100 * rsm.effect_power(v, sigma, n, df, alpha, n - 1)) for k, v in marks.items()]
    st.dataframe(pd.DataFrame(rows, columns=["Factor", "Estimated effect", "Power, unadjusted (%)", "Power, Bonferroni (%)"]).style.format(
        {"Estimated effect": "{:+.3f}", "Power, unadjusted (%)": "{:.0f}", "Power, Bonferroni (%)": "{:.0f}"}), hide_index=True)
    st.caption("Power if the true effect equalled the estimate ('observed power') is a restatement of the p-value. Use it to see the "
               "design's resolution, not as evidence about the effect.")
