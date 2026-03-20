import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model_inference import load_inference_artifacts, preprocess_single
from src.utils import load_params
from fastapi.middleware.cors import CORSMiddleware


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    params = load_params()
    model, scaler, label_encoders, feature_store = load_inference_artifacts(params)
    artifacts["model"] = model
    artifacts["scaler"] = scaler
    artifacts["label_encoders"] = label_encoders
    artifacts["feature_store"] = feature_store
    artifacts["params"] = params
    yield
    artifacts.clear()


app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="Predict 30-day hospital readmission risk for patients.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ dev only — lock this down before production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
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

        return PredictionResponse(
            prediction=pred,
            label="Readmitted" if pred == 1 else "Not Readmitted",
            readmission_probability=round(prob, 4),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

