"""
ML Conversation Chatbot agent for Code-to-Visualizer repository.
Provides a safe conversational interface that can answer code questions, run prototype evaluations via the MLOps dashboard,
and summarize ML experiment metrics. This chatbot is intentionally read-only (no arbitrary shell execution).
"""
import os
import re
import requests
from typing import List, Dict, Any
from pathlib import Path

# Optional LLM integration (OpenAI) — only used for response polishing if available
try:
    from langchain.llms import OpenAI
    LLM_AVAILABLE = True
except Exception:
    OpenAI = None
    LLM_AVAILABLE = False

DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:9000")
REPO_PATH = os.environ.get("REPO_PATH", ".")

CODE_KEYWORDS = ["function", "class", ".py", "import", "def", "return", "how to", "where is", "run", "uvicorn", "server"]
DASH_KEYWORDS = ["dashboard", "metrics", "performance", "accuracy", "f1", "run evaluation", "prototype"]
GOV_KEYWORDS = ["policy", "governance", "compliance", "security", "audit", "monitoring"]


def contains_keywords(text: str, keywords: List[str]) -> bool:
    q = text.lower()
    for k in keywords:
        if k in q:
            return True
    return False


def code_search(query: str, base_path: str = REPO_PATH, max_results: int = 5) -> List[str]:
    base = Path(base_path)
    results = []
    q = query.lower()
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".py", ".md", ".txt", ".html", ".js", ".json", ".yaml", ".yml"}:
            try:
                content = p.read_text(encoding='utf-8')
            except Exception:
                continue
            if q in content.lower() or q in p.name.lower():
                snippet = content[:500].replace('\n', ' ')
                results.append(f"{p.relative_to(base)}: {snippet}")
                if len(results) >= max_results:
                    break
    return results


def list_dashboard_algorithms() -> List[str]:
    try:
        r = requests.get(f"{DASHBOARD_URL}/api/algorithms", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return []
    return []


def run_dashboard_algorithm(name: str) -> Dict[str, Any]:
    try:
        r = requests.post(f"{DASHBOARD_URL}/api/run/{name}", timeout=60)
        if r.status_code == 200:
            return r.json()
        return {"error": f"dashboard returned status {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def synthesize_answer(intent: str, context: Any, query: str) -> str:
    base = ""
    if intent == "code":
        if not context:
            base = "I couldn't find matching code snippets. Try a more specific filename or function name."
        else:
            base = f"Found {len(context)} matches. Top result:\n{context[0]}"
            base += "\n\nRun the local demo (if present): `pip install -r requirements.txt` and `uvicorn agent.server:app --reload --port 8000` in the repo root."
    elif intent == "dashboard":
        lines = ["Prototype performance summary:"]
        for alg, res in context.items():
            if isinstance(res, dict) and res.get('metrics'):
                lines.append(f"- {alg}: {res['metrics']}")
            else:
                lines.append(f"- {alg}: error or no metrics returned: {res}")
        base = "\n".join(lines)
    elif intent == "governance":
        base = context or "Governance KB not found; please add docs/governance-framework.md"
    else:
        base = "Sorry, I couldn't determine how to answer that."

    if LLM_AVAILABLE and os.environ.get("USE_LLM_POLISH", "false").lower() in ("1", "true", "yes"):
        try:
            llm = OpenAI(temperature=0.2)
            prompt = f"You are an assistant. The raw answer is:\n{base}\n\nPolish and shorten for a developer audience."
            polished = llm(prompt)
            return polished
        except Exception:
            return base
    return base


def route_query(query: str) -> Dict[str, Any]:
    if contains_keywords(query, CODE_KEYWORDS):
        intent = "code"
    elif contains_keywords(query, DASH_KEYWORDS):
        intent = "dashboard"
    elif contains_keywords(query, GOV_KEYWORDS):
        intent = "governance"
    else:
        if re.search(r"\b(accuracy|f1|precision|recall|metric|performance)\b", query.lower()):
            intent = "dashboard"
        else:
            intent = "code"

    if intent == "code":
        results = code_search(query, base_path=REPO_PATH, max_results=5)
        answer = synthesize_answer("code", results, query)
        return {"intent": "code", "answer": answer, "results": results}

    if intent == "dashboard":
        algs = list_dashboard_algorithms()
        if not algs:
            ans = "No dashboard or algorithms found at the configured DASHBOARD_URL."
            return {"intent": "dashboard", "answer": ans}
        results = {}
        for a in algs[:3]:
            results[a] = run_dashboard_algorithm(a)
        answer = synthesize_answer("dashboard", results, query)
        return {"intent": "dashboard", "answer": answer, "details": results}

    gov_path = Path(REPO_PATH) / "docs" / "governance-framework.md"
    if gov_path.exists():
        kb = gov_path.read_text(encoding='utf-8')
    else:
        kb = "Governance KB missing."
    answer = synthesize_answer("governance", kb, query)
    return {"intent": "governance", "answer": answer}
