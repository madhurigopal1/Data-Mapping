"""
DBSCAN clustering example
"""
from sklearn.cluster import DBSCAN
from ml_algorithms.template import load_data
from sklearn.metrics import adjusted_rand_score


def train_and_eval():
    X_train, X_test, y_train, y_test = load_data('iris')
    model = DBSCAN(eps=0.5, min_samples=5)
    preds = model.fit_predict(X_test)
    # DBSCAN may label noise as -1; handle gracefully
    try:
        score = adjusted_rand_score(y_test, preds)
    except Exception:
        score = 0.0
    return {"adjusted_rand_score": float(score)}
