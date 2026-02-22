import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import os
import yaml

load_dotenv()

def load_dataset_from_gcs(source: str) -> pd.DataFrame:
    return pd.read_csv(source, index_col=0, low_memory=False)


def fix_duplicate_labels(label_dict):
    seen = {}
    fixed = {}
    for code, label in label_dict.items():
        if label in seen:
            seen[label] += 1
            fixed[code] = f"{label}_{seen[label]}"
        else:
            seen[label] = 0
            fixed[code] = label
    return fixed


def load_fktp_stata(source: str) -> pd.DataFrame:
    with pd.io.stata.StataReader(source) as reader:
        val_labels = reader.value_labels()
        df_fktp_decoded = reader.read(convert_categoricals=False)
        lbllist = reader._lbllist


    fixed_labels = {k: fix_duplicate_labels(v) for k, v in val_labels.items()}

    for col_idx, label_name in enumerate(lbllist):
        if label_name and label_name in fixed_labels:
            col = df_fktp_decoded.columns[col_idx]
            df_fktp_decoded[col] = df_fktp_decoded[col].map(fixed_labels[label_name]).astype('category')

    return df_fktp_decoded


def load_dataset_from_csv(source: Path) -> pd.DataFrame:
    return pd.read_csv(source)


def save_dataframe_to_csv(df: pd.DataFrame, path: Path):
    folder_path = path.parent
    os.makedirs(folder_path, exist_ok=True)
    df.to_csv(path, index=False)


def load_params(param_path: str = "params.yaml") -> dict:
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        raise Exception(f"Error loading parameters: {e}")