# BPJS Hospital Readmission – EDA Dashboard

An interactive Streamlit dashboard for Exploratory Data Analysis of the BPJS hospital readmission dataset.

## Pages

| Page | Description |
|------|-------------|
| 📊 Overview | KPI cards, readmission target distribution, dataset summary |
| 👤 Patient Demographics | Gender, age, marital status, insurance class, patient segment |
| 🏥 Hospital Visits (FKRTL) | Admission trends, visit duration, top diagnoses, severity |
| 🔄 Readmission Analysis | Readmission rates by demographics, time trend, days-between-visits |
| 💰 Cost Analysis | Billing distribution, cost by case group, top expensive diagnoses |
| 🗺️ Geographic Analysis | Province-level visit counts, readmission rates, choropleth map |
| 🩺 Primary Care (FKTP) | Visit trends, top diagnoses, FKTP type, referral patterns |

## Prerequisites

- Python 3.9+
- Cleaned data CSVs placed in `../data/cleaned/`:
  - `peserta.csv`
  - `fkrtl.csv`
  - `fktp_decoded.csv`

## Installation & Running

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Notes

- The dashboard loads data from `../data/cleaned/` relative to `dashboard/app.py`.
- Missing data files are handled gracefully — pages will show a warning instead of crashing.
- All charts are interactive (Plotly). Hover for tooltips, click legend items to filter.
- A `indonesia.geojson` file is expected at the project root for the choropleth map.
