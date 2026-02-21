import pandas as pd
from dotenv import load_dotenv
from utils import load_dataset_from_gcs, save_dataframe_to_csv, load_params
from pathlib import Path

load_dotenv()


def main():
    params = load_params("params.yaml")

    df_peserta = load_dataset_from_gcs(params['peserta_gcs_path'])
    df_fktp = load_dataset_from_gcs(params['fktp_gcs_path'])
    df_fkrtl = load_dataset_from_gcs(params['fkrtl_gcs_path'])

    raw_peserta_path = Path(params['data']['raw_data_folder']) / "peserta.csv"
    raw_fktp_path = Path(params['data']['raw_data_folder']) / "fktp.csv"
    raw_fkrtl_path = Path(params['data']['raw_data_folder']) / "fktl.csv"

    save_dataframe_to_csv(df_peserta, raw_peserta_path)
    save_dataframe_to_csv(df_fktp, raw_fktp_path)
    save_dataframe_to_csv(df_fkrtl, raw_fkrtl_path)


if __name__ == "__main__":
    main()