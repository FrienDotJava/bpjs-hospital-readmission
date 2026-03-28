import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.model_inference import load_inference_artifacts, preprocess_single
from src.utils import load_params
from fastapi.middleware.cors import CORSMiddleware
import os
from apscheduler.schedulers.background import BackgroundScheduler
from api.production_monitoring import run_production_drift_check
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None

class PredictionRequest(BaseModel):
    no_peserta: Optional[int] = None
    tanggal_datang: str
    tanggal_pulang: str
    status_pulang_peserta: str
    kode_nama_diagnosis_primer_ICD10: str
    kepemilikan_perujuk: str
    kode_casemix: str
    jenis_perujuk: str
    tipe_perujuk: str
    kepemilikan_fkrtl: str
    kode_nama_diagnosis_ICD10: str
    kode_INACBGs: str
    tipe_fkrtl: str
    kab_kota_fkrtl: str
    gender: Optional[str] = None
    status_peserta: Optional[str] = None
    segmen_peserta: Optional[str] = None
    provinsi_faskes: Optional[str] = None
    provinsi_tempat_tinggal: Optional[str] = None
    kab_kota_tempat_tinggal: Optional[str] = None
    bobot: Optional[float] = None


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    readmission_probability: float


artifacts = {}

def start_monitoring_job():
    print("Running weekly Evidently AI monitoring...")
    run_production_drift_check()

@asynccontextmanager
async def lifespan(app: FastAPI):
    params = load_params()
    model, scaler, label_encoders, feature_store = load_inference_artifacts(params)
    artifacts["model"] = model
    artifacts["scaler"] = scaler
    artifacts["label_encoders"] = label_encoders
    artifacts["feature_store"] = feature_store
    artifacts["params"] = params

    scheduler = BackgroundScheduler()
    scheduler.add_job(start_monitoring_job, 'interval', weeks=1)
    scheduler.start()
    
    yield

    artifacts.clear()
    scheduler.shutdown()


app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="Predict 30-day hospital readmission risk for patients.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def log_to_supabase(features: dict, pred: int, prob: float):
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

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    try:
        raw_data = request.model_dump(exclude_none=True)
        X = preprocess_single(
            raw_data,
            artifacts["scaler"],
            artifacts["label_encoders"],
            artifacts["feature_store"],
        )

        model = artifacts["model"]
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0, 1])

        features = X.to_dict(orient="records")[0]

        background_tasks.add_task(
            log_to_supabase, 
            features=features, 
            prediction=pred,
            probability=prob
        )
        
        return PredictionResponse(
            prediction=pred,
            label="Readmitted" if pred == 1 else "Not Readmitted",
            readmission_probability=round(prob, 4),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    

@app.get("/report/drift", response_class=HTMLResponse)
def get_production_drift_report():
    file_path = "./reports/production_drift_report.html"
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    else:
        return HTMLResponse(
            content="<html><body><h2>Report not generated yet. Please wait for the scheduler.</h2></body></html>", 
            status_code=404
        )

