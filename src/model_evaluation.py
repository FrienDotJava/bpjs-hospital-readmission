from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from utils import load_params, load_artifact, load_dataset_from_csv, load_json_to_dict
from pathlib import Path
from model_training import split_xy
import json
import os
import mlflow

def save_cm(y_test, y_pred, path):
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, values_format="d")
    disp.plot(cmap="Blues")
    plt.savefig(path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved at {path}")
    mlflow.log_artifact(path)


def get_predictions(model, X_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return preds, probs


def save_metrics(metrics: dict, path: Path):
    folder_path = path.parent
    os.makedirs(folder_path, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved at {path}")


def main():
    params = load_params()
    model_path = params['model_training']['model_path']
    test_data_path = params['data']['test_data_path']
    cm_path = params['model_evaluation']['cm_path']
    metrics_path = params['model_evaluation']['metrics_path']

    model = load_artifact(Path(model_path))

    test = load_dataset_from_csv(Path(test_data_path))
    X_test, y_test = split_xy(test)

    y_pred, probs = get_predictions(model, X_test)

    run_id = load_json_to_dict(Path("misc/latest_run.json"))['run_id']
    with mlflow.start_run(run_id=run_id):
        f1_class_1 = f1_score(y_test, y_pred, average=None)[1]
        precision_class_1 = precision_score(y_test, y_pred)
        recall_class_1 = recall_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        auc = roc_auc_score(y_test, probs)
        
        metrics = {
            "roc_auc": auc,
            "f1_class_1": f1_class_1,
            "precision_class_1": precision_class_1,
            "recall_class_1": recall_class_1,
            "f1_macro": f1_macro,
        }

        save_cm(y_test, y_pred, cm_path)
        save_metrics(metrics, Path(metrics_path))

        mlflow.log_metrics(metrics)
        

if __name__ == "__main__":
    main()
