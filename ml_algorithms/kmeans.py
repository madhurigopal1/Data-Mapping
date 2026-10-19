"""
KMeans clustering example (unsupervised)
"""
from sklearn.cluster import KMeans
from ml_algorithms.template import load_data
from sklearn.metrics import adjusted_rand_score


def train_and_eval():
    X_train, X_test, y_train, y_test = load_data('iris')
    model = KMeans(n_clusters=3, random_state=42)
    preds = model.fit_predict(X_test)
    score = adjusted_rand_score(y_test, preds)
    return {"adjusted_rand_score": float(score)}
