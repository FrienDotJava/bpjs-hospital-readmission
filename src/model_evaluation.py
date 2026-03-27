from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from utils import load_params, load_artifact, load_dataset_from_csv, load_json_to_dict
from pathlib import Path
from model_training import split_xy
import json
import os
import mlflow
from evidently import Report
from evidently.metrics import *
from evidently.presets import *
from evidently import Dataset
from evidently import DataDefinition
from evidently import BinaryClassification

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


def evaluate_and_monitor(model, X_train, y_train, X_test, y_test, params):
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    reference_data = X_train.copy()
    reference_data['target'] = y_train
    reference_data['prediction'] = train_predictions

    current_data = X_test.copy()
    current_data['target'] = y_test
    current_data['prediction'] = test_predictions

    definition = DataDefinition(
        classification=[BinaryClassification(
            target="target",
            prediction_labels="prediction")],
        categorical_columns=["target", "prediction"])

    reference_data = Dataset.from_pandas(
        reference_data,
        data_definition=definition
    )
    current_data = Dataset.from_pandas(
        current_data,
        data_definition=definition
    )

    report = Report([
        DataDriftPreset(), 
        ClassificationPreset()
    ])

    result = report.run(reference_data=reference_data, current_data=current_data)

    report_path = params['model_evaluation']['evidently_report_path']
    result.save_html(report_path)
    print(f"Evidently report generated at: {report_path}")


def main():
    params = load_params()
    model_path = params['model_training']['model_path']
    test_data_path = params['data']['test_data_path']
    train_data_path = params['data']['train_data_path']
    cm_path = params['model_evaluation']['cm_path']
    metrics_path = params['model_evaluation']['metrics_path']

    model = load_artifact(Path(model_path))

    test = load_dataset_from_csv(Path(test_data_path))
    X_test, y_test = split_xy(test)

    train = load_dataset_from_csv(Path(train_data_path))
    X_train, y_train = split_xy(train)

    y_pred, probs = get_predictions(model, X_test)

    evaluate_and_monitor(model, X_train, y_train, X_test, y_test, params)

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
