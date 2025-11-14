from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, get_settings
from .assistant import handle_message
from .db import get_db
from .schemas import ChatRequest, ChatResponse, TodoRead
from .tools.todos import list_todos
from sqlalchemy.orm import Session


app = FastAPI(title="Personal Assistant API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    if settings.api_token and payload.api_token != settings.api_token:
        raise HTTPException(status_code=401, detail="Invalid API token")

    reply = await handle_message(payload.message)
    return reply


@app.get("/todos", response_model=List[TodoRead])
def get_todos(
    status: Optional[str] = None,
    db: Session = Depends(get_db),
) -> List[TodoRead]:
    """Return the list of todos, optionally filtered by status."""
    return list_todos(db, status=status)


