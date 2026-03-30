import streamlit as st
import warnings
import os
import subprocess
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

DATA_FILE = "data/cleaned/peserta.csv"
FASTAPI_URL = "https://bpjs-readmission-api.onrender.com"

if "data_ready" not in st.session_state:
    if not os.path.exists(DATA_FILE):
        with st.spinner("Downloading dataset from DVC..."):
            subprocess.run(["dvc", "pull"], check=True)
    st.session_state["data_ready"] = True

warnings.filterwarnings("ignore")

# Configuration
st.set_page_config(
    page_title="BPJS Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    return prepare_fkrtl(df)


@st.cache_data(show_spinner=False)
def load_fkrtl_summary() -> dict | None:
    path = DATA_DIR / "fkrtl.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_pct": f"{df.isnull().mean().mean() * 100:.1f}%"
    }


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


# Feature engineering
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


# Sidebar
PAGES = [
    "Overview",
    "Patient Demographics",
    "Kunjungan FKRTL",
    "Readmission Analysis",
    "Cost Analysis",
    "Geographic Analysis",
    "Kunjungan FKTP",
    "Model Monitoring"
]

with st.sidebar:
    st.image(
        "https://download.bloguna.com/uploads/322110/BPJS-Kesehatan.png",
        width=150,
    )
    st.title("BPJS EDA Report")
    st.markdown("---")
    selected_page = st.radio("Navigasi", PAGES)
    st.markdown("---")
    st.caption("Data: Sample Data BPJS 2021")


def missing_data_warning(name: str):
    st.warning(
        f"**{name}** data file not found. "
        f"Expected at `{DATA_DIR}`. Some charts may not be available.",
        icon="⚠️",
    )


def kpi_row(metrics: list[dict]):
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            col.metric(m["label"], m["value"], m.get("delta"))


def show_evidently_report(report_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    components.html(html_data, height=1000, scrolling=True)


# PAGE 1 - Overview
if selected_page == PAGES[0]:
    df_peserta = load_peserta()
    df_fkrtl = load_fkrtl()
    df_fktp = load_fktp()
    fkrtl_summary = load_fkrtl_summary()

    st.title("Overview")

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
            {"label": "Total Jumlah Pasien", "value": f"{n_patients:,}" if isinstance(n_patients, int) else n_patients},
            {"label": "Total Kunjungan FKRTL (Rawat Inap)", "value": f"{n_fkrtl:,}" if isinstance(n_fkrtl, int) else n_fkrtl},
            {"label": "Total Kunjungan FKTP", "value": f"{n_fktp:,}" if isinstance(n_fktp, int) else n_fktp},
            {"label": "Tingkat Readmission dalam 30 Hari", "value": readmission_rate_str},
        ]
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Tingkat Readmission")
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
                title="Distribusi Readmission",
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
        if fkrtl_summary is not None:
            summary_rows.append(
                {
                    "Dataset": "FKRTL (Hospital Visits)",
                    "Rows": fkrtl_summary["rows"],
                    "Columns": fkrtl_summary["columns"],
                    "Missing (%)": fkrtl_summary["missing_pct"],
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
            st.info("No data files found.")

    st.subheader("Statistik Fitur Numerik (FKRTL Rawat Inap)")
    if df_fkrtl is not None:
        numeric_cols = ["lama_hari_kunjungan", "jarak_hari_antar_kunjungan", "biaya_tagih", "biaya_verifikasi"]
        available = [c for c in numeric_cols if c in df_fkrtl.columns]
        if available:
            st.dataframe(df_fkrtl[available].describe().round(2), use_container_width=True)
    else:
        missing_data_warning("FKRTL")


# PAGE 2 - Patient Demographics
elif selected_page == PAGES[1]:
    df_peserta = load_peserta()
    st.title("Patient Demographics")
    st.markdown("Analisis Peserta BPJS.")

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
        st.subheader("Distribusi Gender")
        if "gender" in df_peserta.columns:
            fig = bar_chart(df_peserta["gender"], "Distribusi Gender", "Gender")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribusi Status Perkawinan")
        if "status_perkawinan" in df_peserta.columns:
            fig = bar_chart(
                df_peserta["status_perkawinan"],
                "Distribusi Status Perkawinan",
                "Marital Status",
                COLOR_SECONDARY,
            )
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Distribusi Kelas Rawat")
        if "kelas_rawat" in df_peserta.columns:
            fig = bar_chart(
                df_peserta["kelas_rawat"],
                "Kelas Rawat",
                "Jenis Kelas",
                COLOR_SUCCESS,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Distribusi Segmen Peserta")
        if "segmen_peserta" in df_peserta.columns:
            counts = df_peserta["segmen_peserta"].value_counts().reset_index()
            counts.columns = ["segmen_peserta", "count"]
            fig = px.pie(
                counts,
                names="segmen_peserta",
                values="count",
                title="Distribusi Segmen Peserta",
                color_discrete_sequence=CATEGORICAL_PALETTE,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Distribusi Umur
    st.subheader("Distribusi Umur")
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
            title=f"Distribusi Umur)",
            labels={"value": "Umur (tahun)", "count": "Count"},
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Distribusi Status Peserta")
        if "status_peserta" in df_peserta.columns:
            fig = bar_chart(
                df_peserta["status_peserta"],
                "Distribusi Status Peserta",
                "Status Peserta",
                "#9467bd",
            )
            st.plotly_chart(fig, use_container_width=True)

    # with col6:
    #     st.subheader("Facility Ownership Distribution")
    #     if "kepemilikan_faskes" in df_peserta.columns:
    #         fig = bar_chart(
    #             df_peserta["kepemilikan_faskes"],
    #             "Facility Ownership (kepemilikan_faskes)",
    #             "Ownership",
    #             "#8c564b",
    #         )
    #         st.plotly_chart(fig, use_container_width=True)


# PAGE 3 - FKRTL
elif selected_page == PAGES[2]:
    df_fkrtl = load_fkrtl()
    st.title("Kunjungan FKRTL")
    st.markdown("Analisis Kunjungan Faskes Rujukan Tingkat Lanjut.")

    if df_fkrtl is None:
        missing_data_warning("FKRTL")
        st.stop()

    st.subheader("Tren Kunjungan Rawat Inap FKRTL Tahun 2021")
    if "tanggal_datang" in df_fkrtl.columns:
        monthly = (
            df_fkrtl.set_index("tanggal_datang")
            .resample("ME")
            .size()
            .reset_index(name="visits")
        )
        monthly.columns = ["Bulan", "Kunjungan"]
        fig = px.line(
            monthly,
            x="Bulan",
            y="Kunjungan",
            title="Kunjugan Rawat Inap FKRTL Bulanan",
            markers=True,
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Durasi Kunjungan")
        if "lama_hari_kunjungan" in df_fkrtl.columns:
            dur = df_fkrtl["lama_hari_kunjungan"].dropna()
            dur = dur[(dur >= 0) & (dur <= 60)]
            fig = px.histogram(
                dur,
                nbins=60,
                title="Durasi Kunjungan (hari)",
                labels={"value": "Hari", "count": "Count"},
                color_discrete_sequence=[COLOR_SECONDARY],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribusi Kelompok Kasus Rawat Inap")
        if "kelompok_kasus" in df_fkrtl.columns:
            counts = df_fkrtl["kelompok_kasus"].value_counts().reset_index()
            counts.columns = ["kelompok_kasus", "count"]
            fig = px.pie(
                counts,
                names="kelompok_kasus",
                values="count",
                title="Kelompok Kasus",
                color_discrete_sequence=CATEGORICAL_PALETTE,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Top 15 diagnosis
    st.subheader("Top 15 Diagnosis Primer")
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
            title=f"Top 15 Diagnosis Primer",
            color_discrete_sequence=[COLOR_PRIMARY],
            text_auto=True,
        )
        fig.update_layout(yaxis_title="", height=500)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Distribusi Tingkat Keparahan")
        if "tingkat_keparahan_kelompok_kasus" in df_fkrtl.columns:
            counts = df_fkrtl["tingkat_keparahan_kelompok_kasus"].value_counts().reset_index()
            counts.columns = ["severity", "count"]
            fig = px.bar(
                counts,
                x="severity",
                y="count",
                title="Distribusi Tingkat Keparahan",
                color_discrete_sequence=px.colors.sequential.Reds[::-1],
                text_auto=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Distribusi Kepemilikan FKRTL")
        if "kepemilikan_fkrtl" in df_fkrtl.columns:
            counts = df_fkrtl["kepemilikan_fkrtl"].value_counts().reset_index()
            counts.columns = ["kepemilikan", "count"]
            fig = px.bar(
                counts,
                x="kepemilikan",
                y="count",
                title="Kepemilikan FKRTL",
                color_discrete_sequence=[COLOR_SUCCESS],
                text_auto=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribusi Status Pulang")
    if "status_pulang_peserta" in df_fkrtl.columns:
        counts = df_fkrtl["status_pulang_peserta"].value_counts().reset_index()
        counts.columns = ["status", "count"]
        fig = px.bar(
            counts,
            x="status",
            y="count",
            title="Status Pulang",
            color_discrete_sequence=CATEGORICAL_PALETTE,
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)


# PAGE 4 - Readmission Analysis
elif selected_page == PAGES[3]:
    df_peserta = load_peserta()
    df_fkrtl = load_fkrtl()

    st.title("Readmission Analysis")
    st.markdown("Analisis Pola Readmission.")

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
        st.subheader("Tingkat Readmission berdasarkan Gender")
        gender_col = "gender" if "gender" in df_merged.columns else None
        if gender_col:
            fig = readmission_rate_chart(df_merged, gender_col, "Tingkat Readmission berdasarkan Gender")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tingkat Readmission berdasarkan Segmen Peserta")
        fig = readmission_rate_chart(df_merged, "segmen_peserta", "Tingkat Readmission berdasarkan Segmen")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Tingkat Readmission berdasarkan Kelas Rawat")
        kelas_col = "kelas_rawat" if "kelas_rawat" in df_merged.columns else None
        if kelas_col:
            fig = readmission_rate_chart(df_merged, kelas_col, "Tingkat Readmission berdasarkan Kelas Rawat")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Tingkat Readmission berdasarkan Tipe FKRTL")
        fig = readmission_rate_chart(df_merged, "tipe_fkrtl", "Tingkat Readmission berdasarkan Tipe FKRTL")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tingkat Readmission berdasarkan Tingkat Keparahan")
    fig = readmission_rate_chart(
        df_merged,
        "tingkat_keparahan_kelompok_kasus",
        "Tingkat Readmission berdasarkan Tingkat Keparahan",
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tren Tingkat Readmission Tahun 2021")
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
            title="Tingkat Readmission dalam 30 Hari (%)",
            markers=True,
            labels={"rate": "Tingkat Readmission (%)", "month": "Bulan"},
            color_discrete_sequence=[COLOR_DANGER],
        )
        st.plotly_chart(fig, use_container_width=True)


# PAGE 5 - Cost Analysis
elif selected_page == PAGES[4]:
    df_peserta = load_peserta()
    df_fkrtl = load_fkrtl()

    st.title("Cost Analysis")
    st.markdown("Analisis tagihan dan biaya prosedur.")

    if df_fkrtl is None:
        missing_data_warning("FKRTL")
        st.stop()

    cost_cols_available = [c for c in ["biaya_tagih", "biaya_verifikasi"] if c in df_fkrtl.columns]

    if not cost_cols_available:
        st.info("Columns missing in FKRTL data.")
    else:
        st.subheader("Distribusi Biaya Tagih")
        if "biaya_tagih" in df_fkrtl.columns:
            cost = df_fkrtl["biaya_tagih"].dropna()
            cost = cost[cost > 0]
            fig = px.histogram(
                cost,
                nbins=80,
                title="Distribusi Biaya Tagih)",
                labels={"value": "Biaya (IDR)", "count": "Count"},
                color_discrete_sequence=[COLOR_PRIMARY],
                log_y=True,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Rata-Rata Biaya berdasarkan Kelompok Kasus")
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
                    title="Rata-Rata Biaya berdasarkan Kelompok Kasus",
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
                    title="Biaya Tagih berdasarkan Status Readmission",
                    color="Readmitted",
                    color_discrete_map={
                        "Not Readmitted": COLOR_PRIMARY,
                        "Readmitted": COLOR_DANGER,
                    },
                    log_y=True,
                    labels={"biaya_tagih": "Cost (IDR)"},
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top 10 Diagnosis Termahal")
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
                title="Top 10 Diagnosis berdasarkan Biaya Tagih",
                color_discrete_sequence=[COLOR_DANGER],
                text_auto=".2s",
                labels={"avg_cost": "Avg Cost (IDR)"},
            )
            fig.update_layout(height=450, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribusi Tarif Obat")
        if "tarif_drugs" in df_fkrtl.columns:
            drug_cost = df_fkrtl["tarif_drugs"].dropna()
            drug_cost = drug_cost[drug_cost > 0]
            fig = px.histogram(
                drug_cost,
                nbins=60,
                title="Distribusi Tarif Obat",
                labels={"value": "Biaya (IDR)", "count": "Count"},
                color_discrete_sequence=["#9467bd"],
                log_y=True,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column not available in the data.")


# PAGE 6 - Geographic Analysis
elif selected_page == PAGES[5]:
    df_peserta = load_peserta()
    df_fkrtl = load_fkrtl()

    st.title("Geographic Analysis")
    st.markdown("Informasi Kunjungan FKRTL dan Tingkat Readmission pada Tingkat Provinsi.")

    if df_fkrtl is None:
        missing_data_warning("FKRTL")
        st.stop()

    if "provinsi_fkrtl" not in df_fkrtl.columns:
        st.info("Province column not found in FKRTL data.")
        st.stop()

    st.subheader("Jumlah Kunjungan FKRTL berdasarkan Provinsi")
    prov_visits = df_fkrtl["provinsi_fkrtl"].value_counts().reset_index()
    prov_visits.columns = ["province", "visits"]
    fig = px.bar(
        prov_visits.sort_values("visits", ascending=True).tail(25),
        x="visits",
        y="province",
        orientation="h",
        title="Top 25 Provinsi dengan Jumlah Kunjungan Terbanyak",
        color_discrete_sequence=[COLOR_PRIMARY],
        text_auto=True,
    )
    fig.update_layout(height=600, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tingkat Readmission berdasarkan Provinsi")
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
        title="Tingkat Readmission berdasarkan Provinsi",
        color="readmission_rate_pct",
        color_continuous_scale="Reds",
        text_auto=".1f",
        labels={"readmission_rate_pct": "Readmission Rate (%)"},
    )
    fig.update_layout(height=900, yaxis_title="", coloraxis_showscale=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Choropleth Map: Tingkat Readmission berdasarkan Provinsi")
    geojson = load_geojson()
    if geojson is None:
        st.info("Geojson file not found.")
    else:
        try:
            name_key = "state"
            prov_rate_map = prov_rate.copy()

            for i in range(len(geojson['features'])):
                geojson['features'][i]['properties']['state'] = geojson['features'][i]['properties']['state'].upper()
            
            fig = px.choropleth(
                prov_rate_map,
                geojson=geojson,
                locations="province",
                featureidkey=f"properties.{name_key}",
                color="readmission_rate_pct",
                color_continuous_scale="Reds",
                title="Tingkat Readmission berdasarkan Provinsi",
                labels={"readmission_rate_pct": "Tingkat Readmission (%)"},
                fitbounds="locations",
                basemap_visible=False,
            )
            fig.update_layout(height=500, margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render choropleth map: {exc}")


# PAGE 7 - FKTP
elif selected_page == PAGES[6]:
    df_fktp = load_fktp()

    st.title("Kunjungan FKTP")
    st.markdown("Analisis Kunjungan FKTP.")

    if df_fktp is None:
        missing_data_warning("FKTP (fktp_decoded)")
        st.stop()

    st.subheader("Tren Kunjungan FKTP Tahun 2021")
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
            title="Kunjugan FKTP Bulanan",
            markers=True,
            color_discrete_sequence=[COLOR_SUCCESS],
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 15 Diagnosis FKTP")
        diag_col_fktp = "nama_diagnosis" if "nama_diagnosis" in df_fktp.columns else None
        if diag_col_fktp:
            top_diag = df_fktp[diag_col_fktp].value_counts().head(15).reset_index()
            top_diag.columns = ["diagnosis", "count"]
            fig = px.bar(
                top_diag.sort_values("count"),
                x="count",
                y="diagnosis",
                orientation="h",
                title="Top 15 Diagnosis FKTP",
                color_discrete_sequence=[COLOR_SUCCESS],
                text_auto=True,
            )
            fig.update_layout(yaxis_title="", height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Diagnosis column not found in FKTP data.")

    with col2:
        st.subheader("Distribusi Jenis FKTP")
        if "jenis_fktp" in df_fktp.columns:
            counts = df_fktp["jenis_fktp"].value_counts().reset_index()
            counts.columns = ["jenis_fktp", "count"]
            fig = px.pie(
                counts,
                names="jenis_fktp",
                values="count",
                title="Distribusi Jenis FKTP",
                color_discrete_sequence=CATEGORICAL_PALETTE,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column `jenis_fktp` not found.")

    st.subheader("Analisis Pola Referal")
    col3, col4 = st.columns(2)

    with col3:
        if "tingkat_pelayanan" in df_fktp.columns:
            counts = df_fktp["tingkat_pelayanan"].value_counts().reset_index()
            counts.columns = ["tingkat_pelayanan", "count"]
            fig = px.bar(
                counts,
                x="tingkat_pelayanan",
                y="count",
                title="Distribusi Tingkat Pelayanan",
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
                title="Distribusi Jenis Kunjungan",
                color_discrete_sequence=["#8c564b"],
                text_auto=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Status Pulang FKTP")
    if "status_pulang_peserta" in df_fktp.columns:
        counts = df_fktp["status_pulang_peserta"].value_counts().reset_index()
        counts.columns = ["status", "count"]
        fig = px.bar(
            counts,
            x="status",
            y="count",
            title="Status Pulang dari FKTP",
            color_discrete_sequence=CATEGORICAL_PALETTE,
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)


elif selected_page == PAGES[7]:
    st.title("Model Monitoring")

    tab1, tab2 = st.tabs(["Model Evaluation Report", "Production Drift Report"])

    with tab1:
        st.header("Training vs Test Performance")
        show_evidently_report("./reports/evidently_evaluation_report.html")

    with tab2:
        st.header("Production Data Drift")
        try:
            response = requests.get(f"{FASTAPI_URL}/report/drift")
            if response.status_code == 200:
                components.html(response.text, height=1000, scrolling=True)
            elif response.status_code == 404:
                st.warning("The drift report has not been generated yet.")
            else:
                st.error(f"Failed to fetch report. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")
