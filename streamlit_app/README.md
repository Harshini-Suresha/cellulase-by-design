# Response-surface analyser (Streamlit)

Upload a two-factor central composite design (CSV: factor A, factor B, response) and get the quadratic
model, ANOVA with lack of fit, R²/adjusted/predicted R², canonical analysis, optimiser with prediction
intervals, diagnostics and CSV downloads. Ships with the published cellulase CCD as an example.

Three pages that follow the study workflow: generate a design, screen factors, then optimise with a response surface.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy free (Streamlit Community Cloud)
1. Put this folder in a GitHub repository.
2. Go to share.streamlit.io, choose the repo, set the main file to `app.py`, deploy.
3. Add the resulting URL as a link on the website.

## Files
- `pages/2_Design_builder.py`  third page: build a randomised run sheet (central composite k=2–5, Box–Behnken, two-level full factorial, or Plackett–Burman 12/20/24 runs) in real units
- `pages/3_Power.py`  fourth page: power and minimum detectable effect of a screening design (non-central t)
- `pages/1_Screening_designs.py`  second page: Plackett–Burman / fractional-factorial screening (ranked effects, Pareto, Lenth/Bonferroni limits)
- `pages/2_Design_generator.py`  third page: randomised run sheet for CCD, Box–Behnken, full factorial or 12-run Plackett–Burman, with natural-unit settings and CSV download
- `tests/test_rsm.py`  14 checks: published statistics, leave-one-out PRESS, ANOVA identities, rotatability, design sizes (`python tests/test_rsm.py`)
- `designs.py` design generators
- `report.py`  Markdown analysis report (downloadable from the main page)
- `tests.py`   76 self-checks: `python tests.py`
- `rsm.py`   statistics (numpy/scipy only)
- `plots.py` matplotlib figures
- `app.py`   the Streamlit interface

Example data: Microbial Cell Factories 23:11 (2024), doi:10.1186/s12934-023-02286-w.
Centre-point replicates in the example are reported identically, so pure error is zero and lack of fit is shown as undefined.

## Publication figures
`python make_figures.py figures` writes eight white-background, colour-blind-safe figures (PNG 300 dpi, SVG, PDF) into `figures/`.
Figure 1 bar heights are digitised from the published chart; all others are recomputed from the data in `rsm.py`.
