from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import pandas as pd
from loguru import logger

def classification_metrics(y_true, y_pred, output_path=None, run_id=None, model_name=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'report': classification_report(y_true, y_pred, output_dict=True)
    }
    logger.info(f"Evaluation metrics: {metrics}")
    if output_path and model_name and run_id:
        df = pd.DataFrame([metrics])
        df.to_json(f"{output_path}/metrics_{model_name}_{run_id}.json", orient='records', lines=True)
    return metrics
