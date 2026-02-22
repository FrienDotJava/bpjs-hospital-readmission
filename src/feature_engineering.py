import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import featuretools as ft
from utils import load_params, load_csv_parse_date, save_dataframe_to_csv, save_dataframe_to_parquet, save_artifact
from sklearn.model_selection import train_test_split
from pathlib import Path


def filter_rawat_inap(df_fkrtl: pd.DataFrame) -> pd.DataFrame:
    """Filter FKRTL to inpatient cases only and sort by patient and visit date."""
    print("Filtering FKRTL to inpatient cases (rawat inap)...")
    rawat_inap_values = ['Rawat Inap Kebidanan', 'Rawat Inap Bukan Prosedur']
    df_fkrtl = df_fkrtl[df_fkrtl['kelompok_kasus'].isin(rawat_inap_values)].copy()
    df_fkrtl = df_fkrtl.sort_values(by=['no_peserta', 'tanggal_datang'])
    return df_fkrtl


def create_readmission_target(df_fkrtl: pd.DataFrame) -> pd.DataFrame:
    """Create 30-day readmission target and visit-level features on FKRTL."""
    print("Creating readmission target and visit-level features...")
    df_fkrtl['tanggal_kunjungan_berikutnya'] = df_fkrtl.groupby('no_peserta')['tanggal_datang'].shift(-1)

    df_fkrtl['jarak_hari_antar_kunjungan'] = (
        df_fkrtl['tanggal_kunjungan_berikutnya'] - df_fkrtl['tanggal_pulang']
    ).dt.days

    df_fkrtl['readmitted_30d'] = np.where(
        (df_fkrtl['jarak_hari_antar_kunjungan'] <= 30) & (df_fkrtl['jarak_hari_antar_kunjungan'] >= 1),
        1,
        0
    )
    df_fkrtl['jarak_hari_antar_kunjungan'] = df_fkrtl['jarak_hari_antar_kunjungan'].fillna(-1)
    df_fkrtl = df_fkrtl.drop(columns=['tanggal_kunjungan_berikutnya'])

    df_fkrtl['lama_hari_kunjungan'] = (df_fkrtl['tanggal_pulang'] - df_fkrtl['tanggal_datang']).dt.days
    df_fkrtl['jml_kunjungan_fkrtl'] = df_fkrtl.groupby("no_peserta").cumcount()

    return df_fkrtl


def enrich_peserta(df_peserta: pd.DataFrame, df_fkrtl: pd.DataFrame) -> pd.DataFrame:
    """Enrich peserta with aggregated diagnosis count and average visit duration from FKRTL."""
    print("Enriching peserta with FKRTL aggregates...")
    jml_diagnosis = df_fkrtl.groupby("no_peserta")['kode_diagnosis_ICD10'].nunique().reset_index()
    jml_diagnosis.columns = ['no_peserta', 'jml_diagnosis']

    df_peserta = df_peserta.merge(jml_diagnosis, on="no_peserta", how="left")

    avg_hari_kunjungan = df_fkrtl.groupby("no_peserta")['lama_hari_kunjungan'].mean().reset_index()
    avg_hari_kunjungan.columns = ['no_peserta', 'avg_hari_kunjungan_fkrtl']

    df_peserta = df_peserta.merge(avg_hari_kunjungan, on="no_peserta", how="left")

    return df_peserta


def create_fktp_features(df_fktp: pd.DataFrame) -> pd.DataFrame:
    """Create visit interval and visit count features on FKTP."""
    print("Creating FKTP visit features...")
    df_fktp = df_fktp.sort_values(by=['no_peserta', 'tanggal_datang'])

    df_fktp['tanggal_kunjungan_berikutnya'] = df_fktp.groupby('no_peserta')['tanggal_datang'].shift(-1)

    df_fktp['jarak_hari_antar_kunjungan'] = (
        df_fktp['tanggal_kunjungan_berikutnya'] - df_fktp['tanggal_pulang']
    ).dt.days

    df_fktp['jml_kunjungan_fktp'] = df_fktp.groupby("no_peserta").cumcount()

    return df_fktp


def merge_datasets(df_fkrtl: pd.DataFrame, df_peserta: pd.DataFrame, df_fktp: pd.DataFrame) -> pd.DataFrame:
    """Merge FKRTL, peserta, and FKTP into a single DataFrame."""
    print("Merging datasets...")
    df_merged = df_fkrtl.merge(
        df_peserta,
        on=["no_peserta", "bobot", "segmen_peserta"],
        how="left",
        suffixes=("", "_drop")
    )
    df_merged = df_merged.drop(columns=[col for col in df_merged.columns if col.endswith("_drop")])

    df_merged = df_merged.merge(
        df_fktp,
        on="id_kunjungan_fktp",
        how="left",
        suffixes=("", "_drop")
    )
    df_merged = df_merged.drop(columns=[col for col in df_merged.columns if col.endswith("_drop")])

    return df_merged


def build_entityset_and_run_dfs(df_peserta: pd.DataFrame, df_fkrtl: pd.DataFrame, df_fktp: pd.DataFrame, params: dict) -> tuple:
    """Build a featuretools EntitySet and run Deep Feature Synthesis."""
    print("Building EntitySet and running Deep Feature Synthesis...")
    df_fkrtl_2 = df_fkrtl.drop(columns=['jarak_hari_antar_kunjungan'])

    es = ft.EntitySet(id="bpjs_data")

    es = es.add_dataframe(
        dataframe_name="peserta",
        dataframe=df_peserta,
        index="no_peserta"
    )
    es = es.add_dataframe(
        dataframe_name="fkrtl",
        dataframe=df_fkrtl_2,
        index="id_kunjungan_fkrtl",
        time_index="tanggal_datang"
    )
    es = es.add_dataframe(
        dataframe_name="fktp",
        dataframe=df_fktp,
        index="id_kunjungan_fktp",
        time_index="tanggal_datang"
    )

    es = es.add_relationship("peserta", "no_peserta", "fkrtl", "no_peserta")
    es = es.add_relationship("peserta", "no_peserta", "fktp", "no_peserta")

    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="fkrtl",
        agg_primitives=["sum", "mean", "max", "min", "std", "count"],
        trans_primitives=["day", "month", "year", "weekday"],
        max_depth=2
    )
    
    save_dataframe_to_csv(feature_matrix, Path(params['data']['feature_matrix_path']))
    print("Feature matrix saved.")

    return feature_matrix, feature_defs


def remove_correlated_features(feature_matrix: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove features with correlation above the given threshold."""
    print(f"Removing features with correlation > {threshold}...")
    correlation_matrix = feature_matrix.corr(numeric_only=True).abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    to_drop = [column for column in upper_triangle.columns
               if any(upper_triangle[column] > threshold)]

    feature_matrix_reduced = feature_matrix.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} correlated features.")

    return feature_matrix_reduced


def create_feature_store(feature_matrix_reduced: pd.DataFrame, feature_matrix: pd.DataFrame, params: dict):
    """Create and save the peserta-level feature store."""
    FEATURE_STORE_COLS = params['feature_engineering']['feature_store_cols']
    print("Creating peserta feature store...")
    features = feature_matrix_reduced.drop(columns=['no_peserta', 'no_keluarga'])
    cat_cols = features.select_dtypes("category").columns
    num_cols = features.select_dtypes("number").columns
    for col in cat_cols:
        features[col] = features[col].cat.codes

    feature_matrix_peserta = features[FEATURE_STORE_COLS].copy()
    feature_matrix_peserta['no_peserta'] = feature_matrix['no_peserta']
    feature_store_peserta = feature_matrix_peserta.groupby("no_peserta")[FEATURE_STORE_COLS].mean().reset_index()
    feature_store_peserta = feature_store_peserta.fillna(0)

    save_dataframe_to_csv(feature_store_peserta, Path(params['data']['feature_store_path']))
    print("Feature store saved.")


def select_final_features(feature_matrix_reduced: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Select final feature columns and attach the target variable."""
    SELECTED_COLS = params['feature_engineering']['selected_cols']
    print("Selecting final features...")
    final_data = feature_matrix_reduced[SELECTED_COLS].copy()
    final_data['readmitted_30d'] = feature_matrix_reduced['readmitted_30d']

    save_dataframe_to_parquet(final_data, Path(params['data']['final_data_path']))
    print("Final data saved.")

    return final_data


def split_and_scale(final_data: pd.DataFrame, params: dict):
    """Split into train/test, scale numeric features, encode categoricals, and save."""
    print("Splitting, scaling, and encoding data...")

    target_col = params['data']['label_column']
    train, test = train_test_split(
        final_data, test_size=0.2, random_state=42
    )

    num_cols = list(train.select_dtypes("number").columns)
    if target_col in num_cols:
        num_cols.remove(target_col)

    cat_cols = list(train.select_dtypes("category").columns)

    scaler = StandardScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols])
    test[num_cols] = scaler.transform(test[num_cols])

    save_artifact(scaler, Path(params['scaler']['scaler_path']))

    for col in cat_cols:
        train[col] = train[col].cat.codes
        test[col] = test[col].cat.codes

    train = train.fillna(0)
    test = test.fillna(0)

    save_dataframe_to_csv(train, Path(params['data']['train_data_path']))
    save_dataframe_to_csv(test, Path(params['data']['test_data_path']))
    print("Train and test sets saved.")


def main():
    params = load_params()

    df_fkrtl = load_csv_parse_date(Path(params['data']['cleaned_fkrtl_path']), ["tanggal_datang", "tanggal_pulang"])
    df_peserta = load_csv_parse_date(Path(params['data']['cleaned_peserta_path']), ["tanggal_lahir"])
    df_fktp = load_csv_parse_date(Path(params['data']['cleaned_fktp_decoded_path']), ["tanggal_datang", "tanggal_pulang"])

    df_fkrtl = filter_rawat_inap(df_fkrtl)
    df_fkrtl = create_readmission_target(df_fkrtl)
    df_peserta = enrich_peserta(df_peserta, df_fkrtl)
    df_fktp = create_fktp_features(df_fktp)

    merge_datasets(df_fkrtl, df_peserta, df_fktp)

    feature_matrix, _ = build_entityset_and_run_dfs(df_peserta, df_fkrtl, df_fktp, params)

    feature_matrix_reduced = remove_correlated_features(feature_matrix)
    create_feature_store(feature_matrix_reduced, feature_matrix, params)

    final_data = select_final_features(feature_matrix_reduced, params)
    split_and_scale(final_data, params)


if __name__ == "__main__":
    main()