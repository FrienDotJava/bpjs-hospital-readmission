import pandas as pd
from evidently import Report
from evidently.metrics import *
from evidently.presets import *
from evidently import Dataset
from evidently import DataDefinition
from supabase import create_client, Client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None

def run_production_drift_check():
    reference_data = pd.read_csv("./data/processed/train.csv")

    response = supabase.table("inference_logs").select("*").execute()

    features_list = [row["features"] for row in response.data]
    current_data = pd.DataFrame(features_list)

    reference_data = Dataset.from_pandas(
        reference_data,
        data_definition=DataDefinition()
    )
    current_data = Dataset.from_pandas(
        current_data,
        data_definition=DataDefinition()
    )

    drift_report = Report(metrics=[DataDriftPreset()])
    result = drift_report.run(reference_data=reference_data, current_data=current_data)
    
    result.save_html("./reports/production_drift_report.html")

if __name__ == "__main__":
    run_production_drift_check()