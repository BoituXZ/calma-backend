from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Calma AI Microservice")

# Define request schema
class ChatRequest(BaseModel):
    user_id: str
    message: str

# Define response schema
class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def root():
    return {"message": "Calma AI microservice is running 🚀"}

@app.post("/chat", response_model=ChatResponse)
def chat_with_ai(req: ChatRequest):
    # Placeholder AI response (replace later with real model)
    return ChatResponse(reply=f"Echo: {req.message}")
