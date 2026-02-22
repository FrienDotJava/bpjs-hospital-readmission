import pandas as pd
from pathlib import Path
from utils import load_params, load_dataset_from_csv, save_dataframe_to_csv


def fill_empty(df:pd.DataFrame, column: str, value):
    return df[column].fillna(value)


def clean_peserta(df: pd.DataFrame, params: dict):
    print("Cleaning peserta dataset")
    new_column_names = params['data_cleaning']['new_column_names_peserta']
    df.columns = new_column_names
    df['tanggal_lahir'] = pd.to_datetime(df['tanggal_lahir'])
    df['tahun_meninggal'] = df['tahun_meninggal'].astype('Int64')
    df['tahun_meninggal'] = fill_empty(df, 'tahun_meninggal', 0)

    print("Peserta dataset cleaned. Saving...")
    save_dataframe_to_csv(df, Path(params['data']['cleaned_peserta_path']))


def clean_fktp(df: pd.DataFrame, params: dict):
    print("Cleaning FKTP dataset")
    new_column_names = params['data_cleaning']['new_column_names_fktp']
    df.columns = new_column_names
    df['tanggal_datang'] = pd.to_datetime(df['tanggal_datang'])
    df['tanggal_pulang'] = pd.to_datetime(df['tanggal_pulang'])

    df['kode_diagnosis_ICD10'] = fill_empty(df, 'kode_diagnosis_ICD10', '999')

    print("FKTP dataset cleaned. Saving...")
    save_dataframe_to_csv(df, Path(params['data']['cleaned_fktp_encoded_path']))


def clean_fktp_decoded(df: pd.DataFrame, params: dict):
    print("Cleaning decoded FKTP dataset")
    new_column_names = params['data_cleaning']['new_column_names_fktp']
    df.columns = new_column_names

    df['kode_nama_diagnosis_ICD10'] = fill_empty(df, 'kode_nama_diagnosis_ICD10', 'Missing')
    df['kode_diagnosis_ICD10'] = df['kode_diagnosis_ICD10'].replace('', '999')

    print("Decoded FKTP dataset cleaned. Saving...")
    save_dataframe_to_csv(df, Path(params['data']['cleaned_fktp_decoded_path']))


def clean_fkrtl(df: pd.DataFrame, params: dict):
    print("Cleaning FKRTL dataset")
    new_column_name_fkrtl = params['data_cleaning']['new_column_names_fkrtl']
    df.columns = new_column_name_fkrtl
    df['tanggal_datang'] = pd.to_datetime(df['tanggal_datang'])
    df['tanggal_pulang'] = pd.to_datetime(df['tanggal_pulang'])

    df['id_kunjungan_fktp'] = fill_empty(df, 'id_kunjungan_fktp', 'Tidak ada')
    df['jenis_prosedur'] = fill_empty(df, 'jenis_prosedur', 'Missing')
    df['kode_sub_acute'] = fill_empty(df, 'kode_sub_acute', 'Tidak ada')
    df['kode_procedures'] = fill_empty(df, 'kode_procedures', 'Tidak ada')
    df['deskripsi_procedures'] = fill_empty(df, 'deskripsi_procedures', 'Tidak ada')
    df['kode_prosthesis'] = fill_empty(df, 'kode_prosthesis', 'Tidak ada')
    df['deskripsi_prosthesis'] = fill_empty(df, 'deskripsi_prosthesis', 'Tidak ada')
    df['kode_investigation'] = fill_empty(df, 'kode_investigation', 'Tidak ada')
    df['deskripsi_investigation'] = fill_empty(df, 'deskripsi_investigation', 'Tidak ada')
    df['kode_drugs'] = fill_empty(df, 'kode_drugs', 'Tidak ada')
    df['deskripsi_drugs'] = fill_empty(df, 'deskripsi_drugs', 'Tidak ada')

    print("FKRTL dataset cleaned. Saving...")
    save_dataframe_to_csv(df, Path(params['data']['cleaned_fkrtl_path']))

def main():
    params = load_params()

    df_peserta = load_dataset_from_csv(Path(params['data']['raw_peserta_path']))
    df_fktp = load_dataset_from_csv(Path(params['data']['raw_fktp_path']))
    df_fkrtl = load_dataset_from_csv(Path(params['data']['raw_fkrtl_path']))
    df_fktp_decoded = load_dataset_from_csv(Path(params['data']['raw_fktp_decoded_path']))

    clean_peserta(df_peserta, params)

    clean_fktp(df_fktp, params)

    clean_fktp_decoded(df_fktp_decoded, params)
    
    clean_fkrtl(df_fkrtl, params)

if __name__ == "__main__":
    main()