from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from ml_chatbot.agent import route_query

app = FastAPI(title="ML Conversation Chatbot")

class ChatRequest(BaseModel):
    query: str

@app.get("/health")
async def health():
    return {"status":"ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        resp = route_query(req.query)
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8001))
    uvicorn.run(app, host='0.0.0.0', port=port)
