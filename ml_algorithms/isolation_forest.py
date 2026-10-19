"""
Isolation Forest anomaly detection example
"""
from sklearn.ensemble import IsolationForest
from ml_algorithms.template import load_data
from sklearn.metrics import roc_auc_score
import numpy as np


def train_and_eval():
    X_train, X_test, y_train, y_test = load_data('breast_cancer')
    # IsolationForest is unsupervised; create anomaly labels by taking class 1 as anomaly
    model = IsolationForest(random_state=42)
    model.fit(X_train)
    scores = -model.decision_function(X_test)  # higher = more anomalous
    # make binary labels: treat class 1 as positive
    y_binary = (y_test != 0).astype(int)
    try:
        auc = roc_auc_score(y_binary, scores)
    except Exception:
        auc = 0.5
    return {"roc_auc": float(auc)}
