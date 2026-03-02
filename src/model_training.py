import pandas as pd
from xgboost import XGBClassifier
from utils import load_params, save_artifact, load_dataset_from_csv
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, classification_report
from pathlib import Path


def save_cm(y_test, y_pred):
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, values_format="d")
    disp.plot(cmap="Blues")
    plt.savefig("artifacts/confusion_matrix.jpg", format="jpg", dpi=300, bbox_inches="tight")
    plt.close()


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

    model = init_model(model_params)
    train_model(model, X_train, y_train)

    print("Saving model...")
    save_artifact(model,Path(model_path))
    print("Model saved...")


if __name__ == "__main__":
    main()