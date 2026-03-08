"""
BPJS Hospital Readmission - Streamlit EDA Dashboard
Visualizes data from the BPJS hospital readmission prediction project.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BPJS Hospital Readmission EDA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve data paths relative to this file's location (dashboard/) → ../data/cleaned/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "cleaned"
GEOJSON_PATH = BASE_DIR / "indonesia.geojson"

COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#ff7f0e"
COLOR_SUCCESS = "#2ca02c"
COLOR_DANGER = "#d62728"
SEQUENTIAL_PALETTE = px.colors.sequential.Blues
CATEGORICAL_PALETTE = px.colors.qualitative.Set2

RAWAT_INAP_VALUES = ["Rawat Inap Kebidanan", "Rawat Inap Bukan Prosedur"]

# ─────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_peserta() -> pd.DataFrame | None:
    path = DATA_DIR / "peserta.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["tanggal_lahir"], low_memory=False)
    return df


@st.cache_data(show_spinner=False)
def load_fkrtl() -> pd.DataFrame | None:
    path = DATA_DIR / "fkrtl.csv"
    if not path.exists():
        return None
    df = pd.read_csv(
        path,
        parse_dates=["tanggal_datang", "tanggal_pulang"],
        low_memory=False,
    )
    return df


@st.cache_data(show_spinner=False)
def load_fktp() -> pd.DataFrame | None:
    path = DATA_DIR / "fktp_decoded.csv"
    if not path.exists():
        return None
    df = pd.read_csv(
        path,
        parse_dates=["tanggal_datang", "tanggal_pulang"],
        low_memory=False,
    )
    return df


@st.cache_data(show_spinner=False)
def load_geojson():
    if not GEOJSON_PATH.exists():
        return None
    import json

    with open(GEOJSON_PATH, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Feature engineering helpers (mirrors src/feature_engineering.py)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def prepare_fkrtl(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to inpatient, compute lama_hari_kunjungan, jarak_hari_antar_kunjungan, readmitted_30d."""
    df = df[df["kelompok_kasus"].isin(RAWAT_INAP_VALUES)].copy()
    df = df.sort_values(["no_peserta", "tanggal_datang"])

    df["tanggal_kunjungan_berikutnya"] = df.groupby("no_peserta")["tanggal_datang"].shift(-1)
    df["jarak_hari_antar_kunjungan"] = (
        df["tanggal_kunjungan_berikutnya"] - df["tanggal_pulang"]
    ).dt.days
    df["readmitted_30d"] = np.where(
        (df["jarak_hari_antar_kunjungan"] >= 1) & (df["jarak_hari_antar_kunjungan"] <= 30),
        1,
        0,
    )
    df["jarak_hari_antar_kunjungan"] = df["jarak_hari_antar_kunjungan"].fillna(-1)
    df = df.drop(columns=["tanggal_kunjungan_berikutnya"])
    df["lama_hari_kunjungan"] = (df["tanggal_pulang"] - df["tanggal_datang"]).dt.days
    df["jml_kunjungan_fkrtl"] = df.groupby("no_peserta").cumcount()
    return df


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────

PAGES = [
    "📊 Overview",
    "👤 Patient Demographics",
    "🏥 Hospital Visits (FKRTL)",
    "🔄 Readmission Analysis",
    "💰 Cost Analysis",
    "🗺️ Geographic Analysis",
    "🩺 Primary Care (FKTP)",
]

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/BPJS_Kesehatan_logo.svg/320px-BPJS_Kesehatan_logo.svg.png",
        width=150,
    )
    st.title("BPJS Readmission EDA")
    st.markdown("---")
    selected_page = st.radio("Navigate to", PAGES)
    st.markdown("---")
    st.caption("Data: BPJS Hospital Readmission Dataset")

# ─────────────────────────────────────────────
# Load data once (shown with spinner)
# ─────────────────────────────────────────────

with st.spinner("Loading data…"):
    df_peserta = load_peserta()
    df_fkrtl_raw = load_fkrtl()
    df_fktp = load_fktp()

# Prepare engineered FKRTL
df_fkrtl = None
if df_fkrtl_raw is not None:
    df_fkrtl = prepare_fkrtl(df_fkrtl_raw)


def missing_data_warning(name: str):
    st.warning(
        f"⚠️ **{name}** data file not found. "
        f"Expected at `{DATA_DIR}`. Some charts may not be available.",
        icon="⚠️",
    )


# ─────────────────────────────────────────────
# Helper: styled metric row
# ─────────────────────────────────────────────

def kpi_row(metrics: list[dict]):
    """Render a row of KPI metric cards. Each dict has keys: label, value, delta (optional)."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            col.metric(m["label"], m["value"], m.get("delta"))


# ═════════════════════════════════════════════
# PAGE 1 – Overview
# ═════════════════════════════════════════════

if selected_page == PAGES[0]:
    st.title("📊 Overview")
    st.markdown("High-level summary of the BPJS hospital readmission dataset.")

    # KPI metrics
    n_patients = df_peserta["no_peserta"].nunique() if df_peserta is not None else "N/A"
    n_fkrtl = len(df_fkrtl) if df_fkrtl is not None else "N/A"
    n_fktp = len(df_fktp) if df_fktp is not None else "N/A"

    if df_fkrtl is not None:
        readmission_rate = df_fkrtl["readmitted_30d"].mean() * 100
        readmission_rate_str = f"{readmission_rate:.2f}%"
    else:
        readmission_rate_str = "N/A"

    kpi_row(
        [
            {"label": "Total Patients", "value": f"{n_patients:,}" if isinstance(n_patients, int) else n_patients},
            {"label": "Total FKRTL Visits (inpatient)", "value": f"{n_fkrtl:,}" if isinstance(n_fkrtl, int) else n_fkrtl},
            {"label": "Total FKTP Visits", "value": f"{n_fktp:,}" if isinstance(n_fktp, int) else n_fktp},
            {"label": "30-Day Readmission Rate", "value": readmission_rate_str},
        ]
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Readmission distribution pie
    with col1:
        st.subheader("Readmission Target Distribution")
        if df_fkrtl is not None:
            counts = df_fkrtl["readmitted_30d"].value_counts().reset_index()
            counts.columns = ["readmitted_30d", "count"]
            counts["label"] = counts["readmitted_30d"].map({0: "Not Readmitted (0)", 1: "Readmitted (1)"})
            fig = px.pie(
                counts,
                names="label",
                values="count",
                hole=0.45,
                color="label",
                color_discrete_map={
                    "Not Readmitted (0)": COLOR_PRIMARY,
                    "Readmitted (1)": COLOR_DANGER,
                },
                title="30-Day Readmission Class Distribution",
            )
            fig.update_traces(textposition="outside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        else:
            missing_data_warning("FKRTL")

    # Dataset summary
    with col2:
        st.subheader("Dataset Summary")
        summary_rows = []
        if df_peserta is not None:
            summary_rows.append(
                {
                    "Dataset": "Peserta (Members)",
                    "Rows": len(df_peserta),
                    "Columns": len(df_peserta.columns),
                    "Missing (%)": f"{df_peserta.isnull().mean().mean() * 100:.1f}%",
                }
            )
        if df_fkrtl_raw is not None:
            summary_rows.append(
                {
                    "Dataset": "FKRTL (Hospital Visits)",
                    "Rows": len(df_fkrtl_raw),
                    "Columns": len(df_fkrtl_raw.columns),
                    "Missing (%)": f"{df_fkrtl_raw.isnull().mean().mean() * 100:.1f}%",
                }
            )
        if df_fktp is not None:
            summary_rows.append(
                {
                    "Dataset": "FKTP (Primary Care Visits)",
                    "Rows": len(df_fktp),
                    "Columns": len(df_fktp.columns),
                    "Missing (%)": f"{df_fktp.isnull().mean().mean() * 100:.1f}%",
                }
            )
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No data files found. Please place CSV files in `data/cleaned/`.")

    # Numeric summary statistics
    st.subheader("Numeric Feature Statistics (FKRTL inpatient)")
    if df_fkrtl is not None:
        numeric_cols = ["lama_hari_kunjungan", "jarak_hari_antar_kunjungan", "biaya_tagih", "biaya_verifikasi"]
        available = [c for c in numeric_cols if c in df_fkrtl.columns]
        if available:
            st.dataframe(df_fkrtl[available].describe().round(2), use_container_width=True)
    else:
        missing_data_warning("FKRTL")


# ═════════════════════════════════════════════
# PAGE 2 – Patient Demographics
# ═════════════════════════════════════════════

elif selected_page == PAGES[1]:
    st.title("👤 Patient Demographics")
    st.markdown("Analysis of enrolled BPJS members (Peserta).")

    if df_peserta is None:
        missing_data_warning("Peserta")
        st.stop()

    def bar_chart(series: pd.Series, title: str, xlabel: str = "", color: str = COLOR_PRIMARY):
        counts = series.value_counts().reset_index()
        counts.columns = [xlabel or series.name, "count"]
        fig = px.bar(
            counts,
            x=xlabel or series.name,
            y="count",
            title=title,
            color_discrete_sequence=[color],
            text_auto=True,
        )
        fig.update_layout(showlegend=False)
        return fig

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gender Distribution")
        if "gender" in df_peserta.columns:
            fig = bar_chart(df_peserta["gender"], "Gender Distribution", "Gender")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Marital Status Distribution")
        if "status_perkawinan" in df_peserta.columns:
            fig = bar_chart(
                df_peserta["status_perkawinan"],
                "Marital Status Distribution",
                "Marital Status",
                COLOR_SECONDARY,
            )
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Insurance Class Distribution")
        if "kelas_rawat" in df_peserta.columns:
            fig = bar_chart(
                df_peserta["kelas_rawat"],
                "Insurance Class (kelas_rawat)",
                "Insurance Class",
                COLOR_SUCCESS,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Patient Segment Distribution")
        if "segmen_peserta" in df_peserta.columns:
            counts = df_peserta["segmen_peserta"].value_counts().reset_index()
            counts.columns = ["segmen_peserta", "count"]
            fig = px.pie(
                counts,
                names="segmen_peserta",
                values="count",
                title="Patient Segment Distribution",
                color_discrete_sequence=CATEGORICAL_PALETTE,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Age distribution
    st.subheader("Age Distribution")
    if "tanggal_lahir" in df_peserta.columns:
        reference_date = df_peserta["tanggal_lahir"].max() + pd.DateOffset(years=1)
        reference_date = reference_date.replace(month=1, day=1)
        ref_year = reference_date.year
        age = (reference_date - df_peserta["tanggal_lahir"]).dt.days / 365.25
        age = age.dropna()
        age = age[(age >= 0) & (age <= 120)]
        fig = px.histogram(
            age,
            nbins=50,
            title=f"Age Distribution (as of {ref_year})",
            labels={"value": "Age (years)", "count": "Count"},
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Patient Status Distribution")
        if "status_peserta" in df_peserta.columns:
            fig = bar_chart(
                df_peserta["status_peserta"],
                "Patient Status Distribution",
                "Patient Status",
                "#9467bd",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("Facility Ownership Distribution")
        if "kepemilikan_faskes" in df_peserta.columns:
            fig = bar_chart(
                df_peserta["kepemilikan_faskes"],
                "Facility Ownership (kepemilikan_faskes)",
                "Ownership",
                "#8c564b",
            )
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 3 – Hospital Visits (FKRTL)
# ═════════════════════════════════════════════

elif selected_page == PAGES[2]:
    st.title("🏥 Hospital Visits (FKRTL)")
    st.markdown("Analysis of hospital (secondary/tertiary care) inpatient visits.")

    if df_fkrtl is None:
        missing_data_warning("FKRTL")
        st.stop()

    # Monthly admission trends
    st.subheader("Monthly Admission Trends")
    if "tanggal_datang" in df_fkrtl.columns:
        monthly = (
            df_fkrtl.set_index("tanggal_datang")
            .resample("ME")
            .size()
            .reset_index(name="visits")
        )
        monthly.columns = ["month", "visits"]
        fig = px.line(
            monthly,
            x="month",
            y="visits",
            title="Monthly Inpatient Admissions",
            markers=True,
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visit Duration Distribution")
        if "lama_hari_kunjungan" in df_fkrtl.columns:
            dur = df_fkrtl["lama_hari_kunjungan"].dropna()
            dur = dur[(dur >= 0) & (dur <= 60)]
            fig = px.histogram(
                dur,
                nbins=60,
                title="Visit Duration (days)",
                labels={"value": "Days", "count": "Count"},
                color_discrete_sequence=[COLOR_SECONDARY],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Case Group Distribution")
        if "kelompok_kasus" in df_fkrtl.columns:
            counts = df_fkrtl["kelompok_kasus"].value_counts().reset_index()
            counts.columns = ["kelompok_kasus", "count"]
            fig = px.pie(
                counts,
                names="kelompok_kasus",
                values="count",
                title="Case Group (kelompok_kasus)",
                color_discrete_sequence=CATEGORICAL_PALETTE,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Top 15 diagnoses
    st.subheader("Top 15 Primary Diagnoses")
    diag_col = "nama_diagnosis_primer" if "nama_diagnosis_primer" in df_fkrtl.columns else "nama_diagnosis"
    if diag_col in df_fkrtl.columns:
        top_diag = (
            df_fkrtl[diag_col]
            .value_counts()
            .head(15)
            .reset_index()
        )
        top_diag.columns = ["diagnosis", "count"]
        fig = px.bar(
            top_diag.sort_values("count"),
            x="count",
            y="diagnosis",
            orientation="h",
            title=f"Top 15 Primary Diagnoses ({diag_col})",
            color_discrete_sequence=[COLOR_PRIMARY],
            text_auto=True,
        )
        fig.update_layout(yaxis_title="", height=500)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Severity Level Distribution")
        if "tingkat_keparahan_kelompok_kasus" in df_fkrtl.columns:
            counts = df_fkrtl["tingkat_keparahan_kelompok_kasus"].value_counts().reset_index()
            counts.columns = ["severity", "count"]
            fig = px.bar(
                counts,
                x="severity",
                y="count",
                title="Severity Level Distribution",
                color_discrete_sequence=px.colors.sequential.Reds[::-1],
                text_auto=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Hospital Ownership Distribution")
        if "kepemilikan_fkrtl" in df_fkrtl.columns:
            counts = df_fkrtl["kepemilikan_fkrtl"].value_counts().reset_index()
            counts.columns = ["ownership", "count"]
            fig = px.bar(
                counts,
                x="ownership",
                y="count",
                title="Hospital Ownership",
                color_discrete_sequence=[COLOR_SUCCESS],
                text_auto=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Discharge Status Distribution")
    if "status_pulang_peserta" in df_fkrtl.columns:
        counts = df_fkrtl["status_pulang_peserta"].value_counts().reset_index()
        counts.columns = ["status", "count"]
        fig = px.bar(
            counts,
            x="status",
            y="count",
            title="Discharge Status",
            color_discrete_sequence=CATEGORICAL_PALETTE,
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 4 – Readmission Analysis
# ═════════════════════════════════════════════

elif selected_page == PAGES[3]:
    st.title("🔄 Readmission Analysis")
    st.markdown("Deep-dive into 30-day hospital readmission patterns.")

    if df_fkrtl is None:
        missing_data_warning("FKRTL")
        st.stop()

    def readmission_rate_chart(df: pd.DataFrame, group_col: str, title: str):
        if group_col not in df.columns:
            st.info(f"Column `{group_col}` not available.")
            return
        rate = df.groupby(group_col)["readmitted_30d"].mean().reset_index()
        rate.columns = [group_col, "readmission_rate"]
        rate["readmission_rate_pct"] = rate["readmission_rate"] * 100
        rate = rate.sort_values("readmission_rate_pct", ascending=False)
        fig = px.bar(
            rate,
            x=group_col,
            y="readmission_rate_pct",
            title=title,
            labels={"readmission_rate_pct": "Readmission Rate (%)"},
            color="readmission_rate_pct",
            color_continuous_scale="Reds",
            text_auto=".1f",
        )
        fig.update_layout(coloraxis_showscale=False)
        return fig

    # Merge with peserta for demographic dimensions
    df_merged = df_fkrtl.copy()
    if df_peserta is not None:
        peserta_cols = ["no_peserta", "gender", "kelas_rawat"]
        available_peserta = [c for c in peserta_cols if c in df_peserta.columns]
        df_merged = df_fkrtl.merge(
            df_peserta[available_peserta].drop_duplicates("no_peserta"),
            on="no_peserta",
            how="left",
            suffixes=("", "_peserta"),
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Readmission Rate by Gender")
        gender_col = "gender" if "gender" in df_merged.columns else None
        if gender_col:
            fig = readmission_rate_chart(df_merged, gender_col, "Readmission Rate by Gender")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Readmission Rate by Patient Segment")
        fig = readmission_rate_chart(df_merged, "segmen_peserta", "Readmission Rate by Segment")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Readmission Rate by Insurance Class")
        kelas_col = "kelas_rawat" if "kelas_rawat" in df_merged.columns else None
        if kelas_col:
            fig = readmission_rate_chart(df_merged, kelas_col, "Readmission Rate by Insurance Class")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Readmission Rate by Hospital Type")
        fig = readmission_rate_chart(df_merged, "tipe_fkrtl", "Readmission Rate by Hospital Type")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Readmission Rate by Severity Level")
    fig = readmission_rate_chart(
        df_merged,
        "tingkat_keparahan_kelompok_kasus",
        "Readmission Rate by Severity Level",
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Days-between-visits distribution
    st.subheader("Days Between Visits: Readmitted vs. Not Readmitted")
    if "jarak_hari_antar_kunjungan" in df_fkrtl.columns:
        plot_df = df_fkrtl[df_fkrtl["jarak_hari_antar_kunjungan"] > 0].copy()
        plot_df = plot_df[plot_df["jarak_hari_antar_kunjungan"] <= 365]
        plot_df["Readmitted"] = plot_df["readmitted_30d"].map({0: "Not Readmitted", 1: "Readmitted"})
        fig = px.histogram(
            plot_df,
            x="jarak_hari_antar_kunjungan",
            color="Readmitted",
            nbins=60,
            barmode="overlay",
            title="Distribution of Days Between Visits",
            labels={"jarak_hari_antar_kunjungan": "Days Between Visits"},
            color_discrete_map={"Not Readmitted": COLOR_PRIMARY, "Readmitted": COLOR_DANGER},
            opacity=0.75,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Monthly readmission rate trend
    st.subheader("Monthly Readmission Rate Trend")
    if "tanggal_datang" in df_fkrtl.columns:
        monthly = (
            df_fkrtl.set_index("tanggal_datang")
            .resample("ME")["readmitted_30d"]
            .agg(["sum", "count"])
            .reset_index()
        )
        monthly.columns = ["month", "readmitted", "total"]
        monthly["rate"] = monthly["readmitted"] / monthly["total"] * 100
        fig = px.line(
            monthly,
            x="month",
            y="rate",
            title="Monthly 30-Day Readmission Rate (%)",
            markers=True,
            labels={"rate": "Readmission Rate (%)", "month": "Month"},
            color_discrete_sequence=[COLOR_DANGER],
        )
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 5 – Cost Analysis
# ═════════════════════════════════════════════

elif selected_page == PAGES[4]:
    st.title("💰 Cost Analysis")
    st.markdown("Analysis of billing and procedure costs.")

    if df_fkrtl is None:
        missing_data_warning("FKRTL")
        st.stop()

    cost_cols_available = [c for c in ["biaya_tagih", "biaya_verifikasi"] if c in df_fkrtl.columns]

    if not cost_cols_available:
        st.info("No cost columns found in FKRTL data.")
    else:
        # Distribution of billing costs
        st.subheader("Billing Cost Distribution")
        if "biaya_tagih" in df_fkrtl.columns:
            cost = df_fkrtl["biaya_tagih"].dropna()
            cost = cost[cost > 0]
            fig = px.histogram(
                cost,
                nbins=80,
                title="Distribution of Billing Costs (biaya_tagih)",
                labels={"value": "Cost (IDR)", "count": "Count"},
                color_discrete_sequence=[COLOR_PRIMARY],
                log_y=True,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Average Cost by Case Group")
            if "kelompok_kasus" in df_fkrtl.columns and "biaya_tagih" in df_fkrtl.columns:
                avg_cost = (
                    df_fkrtl.groupby("kelompok_kasus")["biaya_tagih"]
                    .mean()
                    .reset_index()
                    .sort_values("biaya_tagih", ascending=False)
                )
                avg_cost.columns = ["kelompok_kasus", "avg_cost"]
                fig = px.bar(
                    avg_cost,
                    x="kelompok_kasus",
                    y="avg_cost",
                    title="Average Billing Cost by Case Group",
                    color_discrete_sequence=[COLOR_SECONDARY],
                    text_auto=".2s",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Cost: Readmitted vs. Not Readmitted")
            if "biaya_tagih" in df_fkrtl.columns:
                cost_comp = df_fkrtl[["biaya_tagih", "readmitted_30d"]].dropna()
                cost_comp["Readmitted"] = cost_comp["readmitted_30d"].map(
                    {0: "Not Readmitted", 1: "Readmitted"}
                )
                fig = px.box(
                    cost_comp,
                    x="Readmitted",
                    y="biaya_tagih",
                    title="Billing Cost by Readmission Status",
                    color="Readmitted",
                    color_discrete_map={
                        "Not Readmitted": COLOR_PRIMARY,
                        "Readmitted": COLOR_DANGER,
                    },
                    log_y=True,
                    labels={"biaya_tagih": "Cost (IDR)"},
                )
                st.plotly_chart(fig, use_container_width=True)

        # Top 10 most expensive diagnoses
        st.subheader("Top 10 Most Expensive Diagnoses")
        diag_col = "nama_diagnosis_primer" if "nama_diagnosis_primer" in df_fkrtl.columns else "nama_diagnosis"
        if diag_col in df_fkrtl.columns and "biaya_tagih" in df_fkrtl.columns:
            top_expensive = (
                df_fkrtl.groupby(diag_col)["biaya_tagih"]
                .mean()
                .nlargest(10)
                .reset_index()
            )
            top_expensive.columns = ["diagnosis", "avg_cost"]
            fig = px.bar(
                top_expensive.sort_values("avg_cost"),
                x="avg_cost",
                y="diagnosis",
                orientation="h",
                title="Top 10 Diagnoses by Average Billing Cost",
                color_discrete_sequence=[COLOR_DANGER],
                text_auto=".2s",
                labels={"avg_cost": "Avg Cost (IDR)"},
            )
            fig.update_layout(height=450, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        # Drug costs
        st.subheader("Drug Costs Distribution")
        if "tarif_drugs" in df_fkrtl.columns:
            drug_cost = df_fkrtl["tarif_drugs"].dropna()
            drug_cost = drug_cost[drug_cost > 0]
            fig = px.histogram(
                drug_cost,
                nbins=60,
                title="Drug Cost Distribution (tarif_drugs)",
                labels={"value": "Cost (IDR)", "count": "Count"},
                color_discrete_sequence=["#9467bd"],
                log_y=True,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Drug cost column (`tarif_drugs`) not available in the data.")


# ═════════════════════════════════════════════
# PAGE 6 – Geographic Analysis
# ═════════════════════════════════════════════

elif selected_page == PAGES[5]:
    st.title("🗺️ Geographic Analysis")
    st.markdown("Province-level distribution of hospital visits and readmission rates.")

    if df_fkrtl is None:
        missing_data_warning("FKRTL")
        st.stop()

    if "provinsi_fkrtl" not in df_fkrtl.columns:
        st.info("Province column (`provinsi_fkrtl`) not found in FKRTL data.")
        st.stop()

    # Province visit count bar chart
    st.subheader("Hospital Visits by Province")
    prov_visits = df_fkrtl["provinsi_fkrtl"].value_counts().reset_index()
    prov_visits.columns = ["province", "visits"]
    fig = px.bar(
        prov_visits.sort_values("visits", ascending=True).tail(25),
        x="visits",
        y="province",
        orientation="h",
        title="Top 25 Provinces by Number of Hospital Visits",
        color_discrete_sequence=[COLOR_PRIMARY],
        text_auto=True,
    )
    fig.update_layout(height=600, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    # Province readmission rate bar chart
    st.subheader("Readmission Rate by Province")
    prov_rate = (
        df_fkrtl.groupby("provinsi_fkrtl")["readmitted_30d"]
        .agg(["mean", "count"])
        .reset_index()
    )
    prov_rate.columns = ["province", "readmission_rate", "visits"]
    prov_rate["readmission_rate_pct"] = prov_rate["readmission_rate"] * 100
    prov_rate = prov_rate[prov_rate["visits"] >= 10].sort_values(
        "readmission_rate_pct", ascending=True
    )
    fig = px.bar(
        prov_rate,
        x="readmission_rate_pct",
        y="province",
        orientation="h",
        title="30-Day Readmission Rate by Province (min 10 visits)",
        color="readmission_rate_pct",
        color_continuous_scale="Reds",
        text_auto=".1f",
        labels={"readmission_rate_pct": "Readmission Rate (%)"},
    )
    fig.update_layout(height=600, yaxis_title="", coloraxis_showscale=True)
    st.plotly_chart(fig, use_container_width=True)

    # Choropleth map
    st.subheader("Choropleth Map: Readmission Rate by Province")
    geojson = load_geojson()
    if geojson is None:
        st.info("GeoJSON file not found. Skipping choropleth map.")
    else:
        # Try to find the feature ID property matching province names
        try:
            feature_props = geojson["features"][0]["properties"] if geojson.get("features") else {}
            id_keys = list(feature_props.keys())

            # Build province → geojson mapping attempt
            # Common key for province name in Indonesia GeoJSON files
            candidate_keys = [k for k in id_keys if "prov" in k.lower() or "name" in k.lower() or "nama" in k.lower()]
            name_key = candidate_keys[0] if candidate_keys else (id_keys[0] if id_keys else None)

            if name_key:
                prov_rate_map = prov_rate.copy()

                fig = px.choropleth(
                    prov_rate_map,
                    geojson=geojson,
                    locations="province",
                    featureidkey=f"properties.{name_key}",
                    color="readmission_rate_pct",
                    color_continuous_scale="Reds",
                    title="Readmission Rate by Province",
                    labels={"readmission_rate_pct": "Readmission Rate (%)"},
                    fitbounds="locations",
                    basemap_visible=False,
                )
                fig.update_layout(height=500, margin={"r": 0, "t": 40, "l": 0, "b": 0})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Could not determine province name field in GeoJSON.")
        except Exception as exc:
            st.warning(f"Could not render choropleth map: {exc}")


# ═════════════════════════════════════════════
# PAGE 7 – Primary Care (FKTP)
# ═════════════════════════════════════════════

elif selected_page == PAGES[6]:
    st.title("🩺 Primary Care (FKTP)")
    st.markdown("Analysis of primary healthcare (FKTP) visits.")

    if df_fktp is None:
        missing_data_warning("FKTP (fktp_decoded)")
        st.stop()

    # Visit trends over time
    st.subheader("Monthly FKTP Visit Trends")
    if "tanggal_datang" in df_fktp.columns:
        monthly_fktp = (
            df_fktp.set_index("tanggal_datang")
            .resample("ME")
            .size()
            .reset_index(name="visits")
        )
        monthly_fktp.columns = ["month", "visits"]
        fig = px.line(
            monthly_fktp,
            x="month",
            y="visits",
            title="Monthly Primary Care Visits",
            markers=True,
            color_discrete_sequence=[COLOR_SUCCESS],
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 15 Primary Care Diagnoses")
        diag_col_fktp = "nama_diagnosis" if "nama_diagnosis" in df_fktp.columns else None
        if diag_col_fktp:
            top_diag = df_fktp[diag_col_fktp].value_counts().head(15).reset_index()
            top_diag.columns = ["diagnosis", "count"]
            fig = px.bar(
                top_diag.sort_values("count"),
                x="count",
                y="diagnosis",
                orientation="h",
                title="Top 15 Diagnoses (FKTP)",
                color_discrete_sequence=[COLOR_SUCCESS],
                text_auto=True,
            )
            fig.update_layout(yaxis_title="", height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Diagnosis column not found in FKTP data.")

    with col2:
        st.subheader("FKTP Type Distribution")
        if "jenis_fktp" in df_fktp.columns:
            counts = df_fktp["jenis_fktp"].value_counts().reset_index()
            counts.columns = ["jenis_fktp", "count"]
            fig = px.pie(
                counts,
                names="jenis_fktp",
                values="count",
                title="FKTP Type Distribution (jenis_fktp)",
                color_discrete_sequence=CATEGORICAL_PALETTE,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column `jenis_fktp` not found.")

    # Referral pattern analysis
    st.subheader("Referral Pattern Analysis")
    col3, col4 = st.columns(2)

    with col3:
        if "tingkat_pelayanan" in df_fktp.columns:
            counts = df_fktp["tingkat_pelayanan"].value_counts().reset_index()
            counts.columns = ["tingkat_pelayanan", "count"]
            fig = px.bar(
                counts,
                x="tingkat_pelayanan",
                y="count",
                title="Level of Service (tingkat_pelayanan)",
                color_discrete_sequence=[COLOR_SECONDARY],
                text_auto=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "jenis_kunjungan_fktp" in df_fktp.columns:
            counts = df_fktp["jenis_kunjungan_fktp"].value_counts().reset_index()
            counts.columns = ["jenis_kunjungan_fktp", "count"]
            fig = px.bar(
                counts,
                x="jenis_kunjungan_fktp",
                y="count",
                title="Visit Type (jenis_kunjungan_fktp)",
                color_discrete_sequence=["#8c564b"],
                text_auto=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    # FKTP discharge status
    st.subheader("FKTP Discharge Status")
    if "status_pulang_peserta" in df_fktp.columns:
        counts = df_fktp["status_pulang_peserta"].value_counts().reset_index()
        counts.columns = ["status", "count"]
        fig = px.bar(
            counts,
            x="status",
            y="count",
            title="Discharge Status at FKTP Level",
            color_discrete_sequence=CATEGORICAL_PALETTE,
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)
