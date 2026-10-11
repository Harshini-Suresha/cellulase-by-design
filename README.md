# Cellulase by Design

Design of experiments and statistical optimisation of Aspergillus awamori MK788209 cellulase from agricultural waste.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harshini-Suresha/cellulase-by-design/blob/main/notebooks/cellulase_by_design.ipynb)

- **Website (GitHub Pages):** https://harshini-suresha.github.io/cellulase-by-design/
  (served from the repo root `index.html` + `img/`; `website/` is a synced copy).
  Open `index.html` locally, or deploy via Settings → Pages → Deploy from branch → `main` / `/ (root)`.
- **Interactive notebook:** [`notebooks/cellulase_by_design.ipynb`](notebooks/cellulase_by_design.ipynb)
  — self-contained (numpy/pandas/scipy/matplotlib only, all preinstalled in Colab).
  Click the Colab badge above, or open in Jupyter/VS Code. Also viewable on
  [nbviewer](https://nbviewer.org/github/Harshini-Suresha/cellulase-by-design/blob/main/notebooks/cellulase_by_design.ipynb).
- **`streamlit_app/`** three-page Streamlit tool (design generator, screening analyser, response-surface analyser) and tests.
  See its README for running and free deployment (Streamlit Community Cloud,
  main file `streamlit_app/app.py`).

Contact: harshinisuresha7@gmail.com

Source data: Microbial Cell Factories 23:11 (2024), doi:10.1186/s12934-023-02286-w.

## Run locally

```bash
# static site
python3 -m http.server 8000   # then open http://localhost:8000

# streamlit app
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py

# notebook
jupyter notebook notebooks/cellulase_by_design.ipynb
```
