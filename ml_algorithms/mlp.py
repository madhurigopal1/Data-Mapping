"""
MLP Neural Network example
"""
from sklearn.neural_network import MLPClassifier
from ml_algorithms.template import load_data, evaluate_model


def train_and_eval():
    X_train, X_test, y_train, y_test = load_data('iris')
    model = MLPClassifier(max_iter=500)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return evaluate_model(y_test, preds)
