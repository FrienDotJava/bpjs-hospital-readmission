import pandas as pd
from xgboost import XGBClassifier
from utils import load_params, save_artifact, load_dataset_from_csv, save_dict_to_json
from pathlib import Path
import dagshub
import mlflow

dagshub.init(repo_owner='FrienDotJava', repo_name='bpjs-hospital-readmission', mlflow=True)

def split_xy(df: pd.DataFrame):
    X = df.drop(columns=['readmitted_30d'])
    y = df[['readmitted_30d']]
    return X, y


def init_model(model_params: dict):
    return XGBClassifier(**model_params, random_state=42, eval_metric='aucpr')


def train_model(model: XGBClassifier, X_train: pd.DataFrame, y_train: pd.DataFrame):
    print("Training model start...")
    model.fit(X_train, y_train, verbose=False)
    print("Training model finished...")


def main():
    params = load_params()
    model_params = params["model"]
    model_path = params['model_training']['model_path']
    train_data_path = params['data']['train_data_path']

    train = load_dataset_from_csv(Path(train_data_path))

    X_train, y_train = split_xy(train)

    mlflow.set_experiment("Final")
    with mlflow.start_run() as run:
        model = init_model(model_params)
        train_model(model, X_train, y_train)

        print("Saving model...")
        save_artifact(model,Path(model_path))
        
        mlflow.xgboost.log_model(
            xgb_model=model, 
            artifact_path="model",
            registered_model_name="Final_XGB"
        )
        print("Model saved...")

        mlflow.log_params(model_params)

        metadata = {"run_id": run.info.run_id}
        save_dict_to_json(metadata, Path("misc/latest_run.json"))


if __name__ == "__main__":
    main()