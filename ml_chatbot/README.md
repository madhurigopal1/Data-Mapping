# ML Conversation Chatbot

This folder contains a small ML-focused conversational chatbot that can:
- answer code-related questions by searching repository files
- query the MLOps dashboard for prototype metrics and run example algorithms
- provide simple governance KB answers from docs/governance-framework.md

How to run (demo):
1. Ensure the MLOps dashboard from the repository is running (see dashboard/ README) on port 9000.
2. Set env vars:
   export DASHBOARD_URL=http://localhost:9000
   export REPO_PATH=/path/to/Code-to-Visualizer  # repo root
3. Install requirements and run the server:
   pip install -r ml_chatbot/requirements.txt
   uvicorn ml_chatbot.server:app --reload --port 8001

Endpoints
- GET /health
- POST /chat {"query":"..."}
