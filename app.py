from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
from chatbot_core import generate_chatbot_reply

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_input: str
    history: List[Message]

class ChatResponse(BaseModel):
    reply: str
    sentiment: str
    confidence: float

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(chat: ChatRequest):
    reply, sentiment, confidence = generate_chatbot_reply(
        user_input=chat.user_input,
        previous_messages=[m.dict() for m in chat.history]
    )
    return ChatResponse(reply=reply, sentiment=sentiment, confidence=confidence)

@app.get("/")
def root():
    return {"message": "Mental Health Chatbot API is running."}