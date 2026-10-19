"""
PCA dimensionality reduction example
"""
from sklearn.decomposition import PCA
from ml_algorithms.template import load_data


def train_and_eval():
    X_train, X_test, y_train, y_test = load_data('iris')
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(X_test)
    return {"explained_variance_ratio": list(map(float, pca.explained_variance_ratio_))}
