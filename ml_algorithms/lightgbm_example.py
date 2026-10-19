"""
LightGBM example (if lightgbm installed)
"""
try:
    import lightgbm as lgb
    has_lgb = True
except Exception:
    has_lgb = False

from ml_algorithms.template import load_data, evaluate_model
from sklearn.ensemble import GradientBoostingClassifier


def train_and_eval():
    X_train, X_test, y_train, y_test = load_data('iris')
    if not has_lgb:
        # fallback to sklearn gradient boosting
        model = GradientBoostingClassifier()
    else:
        model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return evaluate_model(y_test, preds)
