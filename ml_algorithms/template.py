"""
Template trainer to be reused by algorithm-specific example scripts.
Provides common data loading (Iris) and evaluation helpers.
"""
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def load_data(dataset_name: str = "iris"):
    if dataset_name == "iris":
        data = load_iris()
    elif dataset_name == "wine":
        data = load_wine()
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
    else:
        data = load_iris()
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return {"accuracy": float(acc), "f1_macro": float(f1)}
