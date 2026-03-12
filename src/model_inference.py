import pandas as pd
from pathlib import Path
from utils import load_params, load_artifact, load_dataset_from_csv

SCALER_COLS = [
    "peserta.STD(fkrtl.lama_hari_kunjungan)",
    "peserta.SUM(fkrtl.jml_kunjungan_fkrtl)",
    "jml_kunjungan_fkrtl",
    "peserta.SUM(fkrtl.lama_hari_kunjungan)",
    "peserta.SUM(fktp.jarak_hari_antar_kunjungan)",
    "peserta.MEAN(fktp.jarak_hari_antar_kunjungan)",
    "peserta.STD(fkrtl.spesifikasi_kelompok_kasus)",
    "peserta.MAX(fkrtl.tarif_drugs)",
    "peserta.SUM(fktp.no_keluarga)",
    "peserta.MEAN(fkrtl.spesifikasi_kelompok_kasus)",
    "peserta.SUM(fkrtl.biaya_tagih)",
    "peserta.MAX(fktp.jarak_hari_antar_kunjungan)",
    "peserta.SUM(fkrtl.no_keluarga)",
    "peserta.MIN(fkrtl.spesifikasi_kelompok_kasus)",
    "peserta.SUM(fkrtl.bobot)",
    "peserta.STD(fkrtl.biaya_tagih)",
    "peserta.SUM(fktp.bobot)",
    "peserta.MIN(fkrtl.biaya_tagih)",
    "peserta.COUNT(fktp)",
]

PESERTA_CAT_COLS = [
    "peserta.status_peserta",
    "peserta.provinsi_faskes",
    "peserta.segmen_peserta",
    "peserta.kab_kota_tempat_tinggal",
    "peserta.provinsi_tempat_tinggal",
    "peserta.gender",
]

FEATURE_STORE_COLS = [
    "peserta.STD(fkrtl.lama_hari_kunjungan)",
    "peserta.SUM(fkrtl.jml_kunjungan_fkrtl)",
    "peserta.SUM(fkrtl.lama_hari_kunjungan)",
    "peserta.status_peserta",
    "peserta.SUM(fktp.jarak_hari_antar_kunjungan)",
    "peserta.provinsi_faskes",
    "peserta.MEAN(fktp.jarak_hari_antar_kunjungan)",
    "peserta.segmen_peserta",
    "peserta.STD(fkrtl.spesifikasi_kelompok_kasus)",
    "peserta.MAX(fkrtl.tarif_drugs)",
    "peserta.kab_kota_tempat_tinggal",
    "peserta.provinsi_tempat_tinggal",
    "peserta.SUM(fktp.no_keluarga)",
    "peserta.MEAN(fkrtl.spesifikasi_kelompok_kasus)",
    "peserta.SUM(fkrtl.biaya_tagih)",
    "peserta.MAX(fktp.jarak_hari_antar_kunjungan)",
    "peserta.SUM(fkrtl.no_keluarga)",
    "peserta.MIN(fkrtl.spesifikasi_kelompok_kasus)",
    "peserta.SUM(fkrtl.bobot)",
    "peserta.STD(fkrtl.biaya_tagih)",
    "peserta.SUM(fktp.bobot)",
    "peserta.MIN(fkrtl.biaya_tagih)",
    "peserta.gender",
    "peserta.COUNT(fktp)",
]

VISIT_LEVEL_COLS = [
    "jml_kunjungan_fkrtl",
    "status_pulang_peserta",
    "MONTH(tanggal_pulang)",
    "YEAR(tanggal_pulang)",
    "MONTH(tanggal_datang)",
    "kode_nama_diagnosis_primer_ICD10",
    "kepemilikan_perujuk",
    "kode_casemix",
    "jenis_perujuk",
    "tipe_perujuk",
    "kepemilikan_fkrtl",
    "kode_nama_diagnosis_ICD10",
    "kode_INACBGs",
    "tipe_fkrtl",
    "kab_kota_fkrtl",
]

def load_inference_artifacts(params: dict):
    model = load_artifact(Path(params["model_training"]["model_path"]))
    scaler = load_artifact(Path(params["scaler"]["scaler_path"]))

    encoder_path = Path(params["encoder"]["encoder_path"])
    label_encoders = load_artifact(encoder_path)

    feature_store = load_dataset_from_csv(Path(params["data"]["feature_store_path"]))

    return model, scaler, label_encoders, feature_store


def preprocess_single(raw_data: dict, scaler, label_encoders, feature_store):
    params = load_params()
    VISIT_CAT_COLS = params['feature_engineering']['cat_cols']

    tanggal_datang = pd.to_datetime(raw_data["tanggal_datang"])
    tanggal_pulang = pd.to_datetime(raw_data["tanggal_pulang"])

    visit = {
        "jml_kunjungan_fkrtl": raw_data.get("jml_kunjungan_fkrtl", 0),
        "status_pulang_peserta": raw_data["status_pulang_peserta"],
        "MONTH(tanggal_pulang)": tanggal_pulang.month,
        "YEAR(tanggal_pulang)": tanggal_pulang.year,
        "MONTH(tanggal_datang)": tanggal_datang.month,
        "kode_nama_diagnosis_primer_ICD10": raw_data["kode_nama_diagnosis_primer_ICD10"],
        "kepemilikan_perujuk": raw_data["kepemilikan_perujuk"],
        "kode_casemix": raw_data["kode_casemix"],
        "jenis_perujuk": raw_data["jenis_perujuk"],
        "tipe_perujuk": raw_data["tipe_perujuk"],
        "kepemilikan_fkrtl": raw_data["kepemilikan_fkrtl"],
        "kode_nama_diagnosis_ICD10": raw_data["kode_nama_diagnosis_ICD10"],
        "kode_INACBGs": raw_data["kode_INACBGs"],
        "tipe_fkrtl": raw_data["tipe_fkrtl"],
        "kab_kota_fkrtl": raw_data["kab_kota_fkrtl"],
    }

    no_peserta = raw_data.get("no_peserta")
    peserta_row = None
    if no_peserta is not None:
        peserta_row = feature_store[feature_store["no_peserta"] == no_peserta]
        if peserta_row.empty:
            peserta_row = None

    is_new_patient = peserta_row is None

    if not is_new_patient:
        peserta_features = peserta_row.iloc[0].to_dict()
    else:
        peserta_features = _build_new_patient_features(raw_data)

    all_feature_cols = SCALER_COLS + VISIT_CAT_COLS + PESERTA_CAT_COLS
    row: dict = {}
    for col in all_feature_cols:
        if col in visit:
            row[col] = visit[col]
        elif col in peserta_features:
            row[col] = peserta_features[col]
        else:
            row[col] = 0

    df = pd.DataFrame([row])

    cols_to_encode = VISIT_CAT_COLS + (PESERTA_CAT_COLS if is_new_patient else [])
    for col in cols_to_encode:
        le = label_encoders[col]
        val_str = str(df[col].iloc[0])
        if val_str in le.classes_:
            df[col] = le.transform([val_str])[0]
        else:
            print(f"Warning: unseen category '{val_str}' in column '{col}', assigning -1")
            df[col] = -1

    if not is_new_patient:
        for col in PESERTA_CAT_COLS:
            df[col] = df[col].astype(int)

    df[SCALER_COLS] = scaler.transform(df[SCALER_COLS])

    selected_cols = params["feature_engineering"]["selected_cols"]
    df = df[selected_cols]

    return df


def _build_new_patient_features(raw_data: dict) -> dict:
    return {
        "peserta.gender": raw_data.get("gender", "LAKI-LAKI"),
        "peserta.status_peserta": raw_data.get("status_peserta", "AKTIF"),
        "peserta.segmen_peserta": raw_data.get("segmen_peserta", "PPU"),
        "peserta.provinsi_faskes": raw_data.get("provinsi_faskes", "JAWA BARAT"),
        "peserta.provinsi_tempat_tinggal": raw_data.get("provinsi_tempat_tinggal", "JAWA BARAT"),
        "peserta.kab_kota_tempat_tinggal": raw_data.get("kab_kota_tempat_tinggal", "BANDUNG"),
        "peserta.STD(fkrtl.lama_hari_kunjungan)": 0,
        "peserta.SUM(fkrtl.jml_kunjungan_fkrtl)": 0,
        "peserta.SUM(fkrtl.lama_hari_kunjungan)": 0,
        "peserta.SUM(fktp.jarak_hari_antar_kunjungan)": 0,
        "peserta.MEAN(fktp.jarak_hari_antar_kunjungan)": 0,
        "peserta.STD(fkrtl.spesifikasi_kelompok_kasus)": 0,
        "peserta.MAX(fkrtl.tarif_drugs)": 0,
        "peserta.SUM(fktp.no_keluarga)": 0,
        "peserta.MEAN(fkrtl.spesifikasi_kelompok_kasus)": 0,
        "peserta.SUM(fkrtl.biaya_tagih)": 0,
        "peserta.MAX(fktp.jarak_hari_antar_kunjungan)": 0,
        "peserta.SUM(fkrtl.no_keluarga)": 0,
        "peserta.MIN(fkrtl.spesifikasi_kelompok_kasus)": 0,
        "peserta.SUM(fkrtl.bobot)": raw_data.get("bobot", 0),
        "peserta.STD(fkrtl.biaya_tagih)": 0,
        "peserta.SUM(fktp.bobot)": raw_data.get("bobot", 0),
        "peserta.MIN(fkrtl.biaya_tagih)": 0,
        "peserta.COUNT(fktp)": 0,
    }


def predict_single(raw_data: dict):
    params = load_params()
    model, scaler, label_encoders, feature_store = load_inference_artifacts(params)
    X = preprocess_single(raw_data, scaler, label_encoders, feature_store)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0, 1]

    return int(pred), float(prob)


def main():
    print("=" * 50)
    print("Example 1: Existing patient (no_peserta=6368)")
    print("=" * 50)
    existing_patient = {
        "no_peserta": 6368,
        "tanggal_datang": "2021-03-15",
        "tanggal_pulang": "2021-03-18",
        "jml_kunjungan_fkrtl": 2,
        "status_pulang_peserta": "Sehat",
        "kode_nama_diagnosis_primer_ICD10": "E11 Type 2 diabetes mellitus",
        "kepemilikan_perujuk": "PemKab",
        "kode_casemix": "E. Endocrine system, nutrition & metabo",
        "jenis_perujuk": "Rumah sakit",
        "tipe_perujuk": "RS KELAS B",
        "kepemilikan_fkrtl": "Pemerintah kab/kota",
        "kode_nama_diagnosis_ICD10": "E11 Type 2 diabetes mellitus",
        "kode_INACBGs": "E-4-10-I",
        "tipe_fkrtl": "RS Kelas B",
        "kab_kota_fkrtl": "TULANGBAWANG",
    }
    pred, prob = predict_single(existing_patient)
    print(f"Prediction: {'Readmitted' if pred == 1 else 'Not Readmitted'}")
    print(f"Probability of readmission: {prob:.4f}")

    print()
    print("=" * 50)
    print("Example 2: New patient (no prior visit history)")
    print("=" * 50)
    new_patient = {
        "tanggal_datang": "2021-06-01",
        "tanggal_pulang": "2021-06-05",
        "status_pulang_peserta": "Sehat",
        "kode_nama_diagnosis_primer_ICD10": "J18 Pneumonia, organism unspecified",
        "kepemilikan_perujuk": "Swasta",
        "kode_casemix": "J. Respiratory system Groups",
        "jenis_perujuk": "Rumah sakit",
        "tipe_perujuk": "RS KELAS C",
        "kepemilikan_fkrtl": "Swasta",
        "kode_nama_diagnosis_ICD10": "J18 Pneumonia, organism unspecified",
        "kode_INACBGs": "J-4-16-I",
        "tipe_fkrtl": "RS Kelas C",
        "kab_kota_fkrtl": "SIDOARJO",
        "gender": "PEREMPUAN",
        "status_peserta": "AKTIF",
        "segmen_peserta": "PPU",
        "provinsi_faskes": "JAWA TIMUR",
        "provinsi_tempat_tinggal": "JAWA TIMUR",
        "kab_kota_tempat_tinggal": "SIDOARJO",
        "bobot": 14.5,
    }
    pred, prob = predict_single(new_patient)
    print(f"Prediction: {'Readmitted' if pred == 1 else 'Not Readmitted'}")
    print(f"Probability of readmission: {prob:.4f}")


if __name__ == "__main__":
    main()
