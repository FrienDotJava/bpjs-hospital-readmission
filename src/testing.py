import sys
from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_inference import load_inference_artifacts
from utils import load_params, load_dataset_from_csv
from model_training import split_xy
from supabase import create_client, Client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None

params = load_params()
model, _, _, _ = load_inference_artifacts(params)

test = load_dataset_from_csv("./data/processed/test.csv")
X, _ = split_xy(test)
X = X.head(1)
pred = int(model.predict(X)[0])
prob = float(model.predict_proba(X)[0, 1])

features = X.to_dict(orient="records")[0]

# print(features)
if supabase:
    try:
        data = {
            "features": features,
            "prediction": pred,
            "probability": prob
        }
        supabase.table("inference_logs").insert(data).execute()
        print("Log saved to Supabase successfully.")
    except Exception as e:
        print(f"Failed to log to Supabase: {e}")