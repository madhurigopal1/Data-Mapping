"""
XGBoost example (if xgboost installed)
"""
try:
    from xgboost import XGBClassifier
    has_xgb = True
except Exception:
    has_xgb = False

from ml_algorithms.template import load_data, evaluate_model


def train_and_eval():
    if not has_xgb:
        return {"error": "xgboost not installed"}
    X_train, X_test, y_train, y_test = load_data('iris')
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return evaluate_model(y_test, preds)
