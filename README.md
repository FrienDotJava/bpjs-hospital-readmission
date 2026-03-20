# Hospital Readmission Prediction

A machine learning project predicting 30-day hospital readmission rates for BPJS (Indonesia's national health insurance) patients. Built with a production-grade ML pipeline, interactive dashboards, and web applications for model inference.

## 📋 Project Overview

Hospital readmission is a critical healthcare indicator that impacts patient outcomes and healthcare costs. This project develops a predictive model using historical patient data from BPJS to identify high-risk patients who are likely to be readmitted within 30 days of discharge.

### Key Objectives
- Build a predictive model for 30-day readmission risk
- Analyze patient demographics, visit patterns, and clinical factors
- Identify key drivers of readmission
- Provide accessible interfaces for model inference and data exploration

### Dataset
- **Source**: Data Sample BPJS (Badan Penyelenggara Jaminan Sosial)
- **Data Types**: 
  - **Peserta** (Patient Demographics): Age, gender, marital status, insurance class, location
  - **FKTP** (Primary Care Data): Outpatient visits, diagnoses, referral patterns
  - **FKRTL** (Hospital Inpatient Data): Hospital admissions, discharge status, diagnoses, length of stay
- **Target Variable**: `readmitted_30d` (Binary: readmission within 30 days)
- **Time Period**: 2021 data

---

## 🚀 Live Applications

Interact with the model and explore the data through these applications:

| Application | URL | Type |
|------------|-----|------|
| **EDA Dashboard** | [bpjs-eda-report.streamlit.app](https://bpjs-eda-report.streamlit.app/) | Streamlit |
| **Model Inference App** | [bpjs-next-app.vercel.app](https://bpjs-next-app.vercel.app/) | Next.js |

### Dashboard Features

#### Streamlit Dashboard
- **Overview**: KPI cards, readmission distribution, dataset statistics
- **Patient Demographics**: Gender, age, marital status, insurance class, patient segments
- **Hospital Visits (FKRTL)**: Admission trends, visit duration, top diagnoses, severity analysis
- **Readmission Analysis**: Readmission rates by patient segments, temporal trends
- **Cost Analysis**: Billing distribution, cost by diagnosis, high-cost cases
- **Geographic Analysis**: Province-level visit counts, readmission heatmap with choropleth
- **Primary Care (FKTP)**: Visit trends, referral patterns, FKTP utilization

#### Next.js Application
- Real-time model predictions
- Patient risk scoring
- Individual prediction explanations
- Responsive UI for mobile and desktop use

---

## 🏗️ Project Architecture

### Directory Structure
```
hospital_readmission_prediction/
├── data/
│   ├── raw/                    # Original raw data files
│   ├── cleaned/                # Cleaned and standardized data
│   ├── interim/                # Intermediate processing files
│   ├── processed/              # Final train/test splits
│   └── feature_store/          # Feature matrices
├── src/
│   ├── data_loading.py         # Load data from GCS
│   ├── data_validation.py      # Great Expectations validation
│   ├── data_cleaning.py        # Standardize column names, handle missing values
│   ├── feature_engineering.py  # Create features, scaling, encoding
│   ├── model_training.py       # Train XGBoost classifier
│   ├── model_evaluation.py     # Evaluate metrics, confusion matrix
│   ├── model_inference.py      # Model prediction pipeline
│   └── utils.py                # Helper functions
├── notebooks/
│   ├── 00_data_loading.ipynb           # Load and explore raw data
│   ├── 01_data_validation.ipynb         # Data quality checks
│   ├── 02_data_cleaning.ipynb           # Data preprocessing
│   ├── 03_eda.ipynb                     # Exploratory data analysis
│   ├── 04_feature_engineering.ipynb     # Feature creation & engineering
│   ├── 05_baseline_models.ipynb         # Baseline model comparison
│   ├── 06_xgboost_optimization.ipynb    # Hyperparameter tuning
│   ├── 07_imbalance_handling.ipynb      # Class imbalance techniques
│   └── 08_business_metrics.ipynb        # Business impact & metrics
├── dashboard/              # Streamlit dashboard
├── gx/                     # Great Expectations validation configs
├── artifacts/              # Model artifacts (model, scaler, encoders, metrics)
├── api/                    # FastAPI inference server
├── dvc.yaml               # DVC pipeline configuration
├── params.yaml            # Model parameters and paths
└── requirements.txt       # Python dependencies
```

### ML Pipeline (DVC)

The project uses **Data Version Control (DVC)** to orchestrate the ML pipeline:

```
data_loading → data_validation → data_cleaning → feature_engineering → model_training → model_evaluation
```

**Pipeline Stages:**
1. **data_loading**: Fetch raw data from Google Cloud Storage (GCS)
2. **data_validation**: Validate data quality using Great Expectations
3. **data_cleaning**: Standardize column names, handle missing values
4. **feature_engineering**: Create features, split train/test by time, scale & encode
5. **model_training**: Train XGBoost classifier with optimized hyperparameters
6. **model_evaluation**: Generate confusion matrix and metrics

---

## 🔧 Tech Stack

### Core ML & Data Processing
- **XGBoost**: Gradient boosting classifier for binary readmission prediction
- **scikit-learn**: Preprocessing, scaling, encoding, model evaluation
- **pandas & NumPy**: Data manipulation and numerical computing
- **Featuretools**: Automated feature engineering (ETL)

### Data Quality & Validation
- **Great Expectations**: Data quality validation rules and profiling
- **pandas-profiling**: Data profiling and exploratory analysis

### Experiment Tracking & Workflow
- **DVC (Data Version Control)**: ML pipeline orchestration and reproducibility
- **MLflow**: Experiment tracking and model versioning
- **Optuna**: Hyperparameter optimization with Bayesian search
- **dagshub**: Centralized experiment management

### Class Imbalance Handling
- **imbalanced-learn (imblearn)**: SMOTE, undersampling, oversampling techniques

### Feature Interpretation
- **SHAP**: Shapley Additive exPlanations for model interpretability

### Dashboards & Applications
- **Streamlit**: Interactive EDA dashboard and data exploration
- **Next.js**: Modern web app for model inference and predictions
- **Plotly**: Interactive visualizations and charts

### Cloud & Infrastructure
- **Google Cloud Storage (GCS)**: Data storage and retrieval
- **Vercel**: Hosting for Next.js application
- **Streamlit Cloud**: Hosting for Streamlit dashboard

### API & Backend
- **FastAPI**: High-performance Python API for model serving
- **Uvicorn**: ASGI web server

### Geospatial Analysis
- **Geopandas**: Geographic data analysis for province-level insights

---

## 📊 Methodology

### Feature Engineering
The project creates two types of features:

1. **Temporal Features**: 
   - Time-based train/test split (80% training cutoff)
   - Days between consecutive visits
   - Visit duration (length of stay)
   - Visit frequency per patient

2. **Patient-Level Features**:
   - Demographics: age, gender, marital status
   - Insurance & socioeconomic: insurance class, patient segment
   - Healthcare utilization: visit counts, top diagnoses, primary care patterns
   - Geographic factors: province, district

3. **Encoding & Scaling**:
   - LabelEncoder for categorical variables
   - StandardScaler for numerical features
   - Artifacts saved for production inference

### Model Selection: XGBoost
**Why XGBoost?**
- Handles both numerical and categorical features effectively
- Captures non-linear relationships and feature interactions
- Robust to class imbalance with `scale_pos_weight` parameter
- Fast training and inference
- Built-in feature importance for interpretability
- Empirically proven performance on healthcare datasets

### Hyperparameter Optimization
Tuned using Optuna with cross-validation:
- `max_depth`: Tree depth control
- `learning_rate`: Gradient boosting step size
- `n_estimators`: Number of boosting rounds
- `scale_pos_weight`: Class weight for imbalanced data
- Regularization: `reg_alpha`, `reg_lambda`, `gamma`

### Evaluation Metrics
- **Precision & Recall**: Handle class imbalance
- **AUC-PR**: Focus on minority class (readmitted patients)
- **Confusion Matrix**: TP, FP, FN, TN analysis
- **Business Metrics**: Cost impact, patient risk segmentation

---

## ⚡ Key Challenges & Solutions

### 1. **Class Imbalance**
**Challenge**: Readmission events are rare (~10-15% of cases), causing the model to be biased toward negative class.

**Solutions Implemented**:
- `scale_pos_weight` in XGBoost to penalize false negatives
- SMOTE (Synthetic Minority Over-sampling Technique) for balanced training sets
- AUC-PR metric instead of accuracy (more sensitive to minority class)
- Stratified cross-validation to maintain class distribution

### 2. **Data Integration & Linking**
**Challenge**: Three separate datasets (Peserta, FKTP, FKRTL) with complex relationships via patient ID and visit IDs.

**Solutions Implemented**:
- Multi-step data cleaning with consistent column naming across files
- Temporal alignment of records by patient visit dates
- Feature engineering using Featuretools for automated multi-table feature creation
- Data validation rules to detect linking errors and missing values

### 3. **Temporal Data Leakage**
**Challenge**: Using future information to predict current events would violate causality.

**Solutions Implemented**:
- Time-based train/test split (80% quantile cutoff on discharge dates)
- Careful feature engineering to only use historical information
- No future diagnosis codes or visit data in feature set

### 4. **Missing Data & Sparse Features**
**Challenge**: Many patients have incomplete healthcare records, sparse diagnoses, or missing demographics.

**Solutions Implemented**:
- Systematic imputation strategies (mean for numerical, mode for categorical)
- Feature selection to remove low-variance or highly sparse features
- Great Expectations validation to track data quality metrics
- Robust encoding for missing categorical values

### 5. **Geographic Heterogeneity**
**Challenge**: Readmission rates vary significantly across provinces due to healthcare infrastructure differences.

**Solutions Implemented**:
- Province-level analysis in EDA dashboard
- Geographic features (province, district) in model
- Stratified evaluation by region to ensure model fairness

### 6. **Model Interpretability & Trust**
**Challenge**: Healthcare models must be explainable for clinical adoption.

**Solutions Implemented**:
- XGBoost feature importance scores
- SHAP analysis for individual prediction explanations
- Confusion matrix visualization for clinical threshold validation
- Business metrics dashboard showing patient risk segments

### 7. **Production Model Serving**
**Challenge**: Making predictions accessible to non-technical users.

**Solutions Implemented**:
- FastAPI backend for efficient inference
- Streamlit dashboard for data exploration without coding
- Next.js web app for modern, responsive UI
- Model artifacts (scaler, encoders) stored for reproducible predictions

### 8. **Reproducibility & Pipeline Management**
**Challenge**: ML pipelines are complex with many interdependent steps; easy to break reproducibility.

**Solutions Implemented**:
- DVC for pipeline orchestration and versioning
- params.yaml for centralized configuration
- MLflow for experiment tracking and model registry
- Documented dependency graph (dvc.yaml)
- Version-controlled code and notebooks

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Google Cloud credentials (for data loading from GCS)
- Git & DVC

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/hospital_readmission_prediction.git
   cd hospital_readmission_prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   # or
   source venv/bin/activate      # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up credentials** (if using cloud data)
   ```bash
   # Add your GCS service account key
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-key.json"
   ```

### Running the ML Pipeline

```bash
# Run entire DVC pipeline
dvc repro

# Or run individual stages
dvc repro --single-stage data_loading
dvc repro --single-stage data_validation
# ... etc

# View pipeline DAG
dvc dag
```

### Running the Streamlit Dashboard (Locally)

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### Setting Up the FastAPI Server

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API documentation: `http://localhost:8000/docs`

---

## 📈 Model Performance

**XGBoost Classifier Results** (on test set):
- **Accuracy**: Evaluation metrics available in `artifacts/metrics.json`
- **AUC-PR**: Focus metric for imbalanced readmission prediction
- **Confusion Matrix**: Visualized in `artifacts/confusion_matrix.jpg`

View detailed metrics:
```bash
cat artifacts/metrics.json
```

---

## 🔍 Exploratory Data Analysis

Key insights from EDA:
- Readmission rates vary by patient demographics and geographic location
- Specific diagnoses show higher readmission risk
- Length of stay correlates with readmission probability
- Seasonal and temporal patterns in hospital utilization
- Geographic disparities in healthcare access and readmission rates

See [dashboard README](dashboard/README.md) for interactive exploration.

---

## 📝 Notebooks

The `notebooks/` folder contains a complete narrative of the project:

| Notebook | Purpose |
|----------|---------|
| `00_data_loading.ipynb` | Load and explore raw data from GCS |
| `01_data_validation.ipynb` | Data quality checks and profiling |
| `02_data_cleaning.ipynb` | Standardize names, handle missing values |
| `03_eda.ipynb` | Statistical analysis and visualizations |
| `04_feature_engineering.ipynb` | Create and engineer features |
| `05_baseline_models.ipynb` | Compare baseline models (Logistic, RF, XGB) |
| `06_xgboost_optimization.ipynb` | Hyperparameter tuning with Optuna |
| `07_imbalance_handling.ipynb` | Test SMOTE and resampling techniques |
| `08_business_metrics.ipynb` | Business impact and risk segmentation |

---

## 🔧 Configuration

### Parameters
All configuration is in `params.yaml`:
- Data paths and GCS locations
- Model hyperparameters (learning_rate, max_depth, etc.)
- Preprocessing parameters (scaling, encoding)
- Pipeline stage configurations

### Environment Variables
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to GCS service account key
- `MLFLOW_TRACKING_URI`: MLflow server URL (optional)
- `DAGSHUB_USER_TOKEN`: DagHub token for experiment tracking (optional)

---

## 📊 Great Expectations Validation

Data quality is ensured through Great Expectations configurations:

```bash
# Run validation
great_expectations checkpoint run checkpoint_fkrtl
great_expectations checkpoint run checkpoint_fktp
great_expectations checkpoint run checkpoint_peserta

# View validation reports
open gx/uncommitted/data_docs/local_site/index.html
```

### Validation Rules
- Column presence and data types
- Missing value thresholds
- Statistical bounds on numerical columns
- Categorical value whitelist

---

## 🎯 Use Cases

1. **Clinical Risk Stratification**: Identify high-risk patients for early intervention programs
2. **Resource Planning**: Allocate beds and staff based on readmission forecasts
3. **Quality Improvement**: Monitor readmission trends by facility and provider
4. **Cost Optimization**: Target interventions to reduce costly readmissions
5. **Policy Analytics**: Assess healthcare policies' impact on readmission rates

---

## 📄 License & Attribution

This project uses 2021 BPJS sample data for research and educational purposes.

---

## 📚 References & Resources

- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **DVC ML Pipeline Docs**: https://dvc.org/
- **Great Expectations**: https://greatexpectations.io/
- **Optuna Hyperparameter Tuning**: https://optuna.org/
- **SHAP Interpretability**: https://shap.readthedocs.io/
- **Streamlit Apps**: https://streamlit.io/
- **Healthcare ML Best Practices**: https://arxiv.org/abs/2006.04185

---

**Last Updated**: March 2026
