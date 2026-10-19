"""
Model runner tool for the ML Conversation Chatbot.
It can import a module from ml_algorithms and call train_and_eval().
"""
import importlib
from typing import Dict, Any


def run_algorithm(module_name: str) -> Dict[str, Any]:
    try:
        mod = importlib.import_module(f"ml_algorithms.{module_name}")
    except Exception as e:
        return {"error": f"import failed: {e}"}
    if not hasattr(mod, 'train_and_eval'):
        return {"error": "module missing train_and_eval()"}
    try:
        metrics = mod.train_and_eval()
        return {"metrics": metrics}
    except Exception as e:
        return {"error": str(e)}
