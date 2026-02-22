import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import os
import yaml

load_dotenv()

def load_dataset_from_gcs(source: str) -> pd.DataFrame:
    return pd.read_csv(source, index_col=0, low_memory=False)


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