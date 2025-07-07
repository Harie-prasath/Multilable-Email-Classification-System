from sklearn.metrics import classification_report
import numpy as np

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> str:
    """Print and return classification report for a label."""
    report = classification_report(y_true, y_pred)
    print(f"\nEvaluation for {label}:\n{report}")
    return report
