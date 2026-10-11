"""Streamlit page: analyse a two-level screening design (Plackett-Burman or fractional factorial)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

import rsm
import plots

st.set_page_config(page_title="Screening analyser", page_icon="🔎", layout="wide")
st.markdown("<style>.stApp{background:#FFE4EC;color:#2A1520}</style>", unsafe_allow_html=True)
st.title("Screening-design analyser")
st.caption("Which of many factors matter? Upload a two-level design (Plackett–Burman or fractional factorial) "
           "and get ranked main effects, a Pareto chart and the significance limits.")

with st.sidebar:
    src = st.radio("Source", ["Published example (12-run Plackett–Burman, 11 factors)", "Upload CSV"])
    if src.startswith("Upload"):
        up = st.file_uploader("CSV: one column per factor, response in the last column", type="csv")
        df = pd.read_csv(up) if up else None
        st.caption("Natural units are fine: each column's two levels are detected automatically.")
    else:
        df = rsm.PBD_SAMPLE.copy()

if df is None:
    st.info("Upload a CSV, or use the example, to begin.")
    st.stop()
if df.shape[1] < 3:
    st.error("Need at least two factor columns and one response column.")
    st.stop()

y_col = st.sidebar.selectbox("Response column", list(df.columns), index=df.shape[1] - 1)
fac = [c for c in df.columns if c != y_col]
d = df.apply(pd.to_numeric, errors="coerce").dropna()
bad = [c for c in fac if d[c].nunique() != 2]
if bad:
    st.error("These columns do not have exactly two levels, so they are not screening factors: " + ", ".join(map(str, bad)))
    st.stop()
if len(d) < len(fac) + 1:
    st.error("A screening design needs at least as many runs as factors plus one.")
    st.stop()

tab, info = rsm.screening_effects(rsm.to_levels(d[fac]), d[y_col].to_numpy(float), fac)
l, r = st.columns([3, 2])
with l:
    plots.show(st, plots.pareto(tab, info))
with r:
    st.metric("Runs / factors", f"{len(d)} / {len(fac)}")
    st.metric("Error method", info["method"])
    st.write(f"t-limit **{info['tlim']:.3f}**, Bonferroni limit **{info['blim']:.3f}** (df = {info['df']:.2f}).")
    keep = tab[tab["Verdict"] != "not significant"]["Factor"].tolist()
    st.success("Carry forward to optimisation: " + (", ".join(keep) if keep else "none cleared the t-limit; consider more runs or a wider factor range."))
st.dataframe(tab.style.format({"Effect": "{:+.3f}", "|t|": "{:.2f}", "p": "{:.4f}"}))
st.caption("Effect = mean response at the high level minus mean at the low level. In a saturated design "
           "(runs = factors + 1) there is no error term, so Lenth's pseudo standard error is used; its limits are approximate. "
           "Screening effects assume interactions are negligible; confirm the chosen factors with a response-surface design.")
st.download_button("Download effects (CSV)", tab.to_csv(index=False).encode(), "screening_effects.csv", "text/csv")
