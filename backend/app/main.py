from typing import List, Optional
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .assistant import handle_message
from .db import get_db
from .schemas import ChatRequest, ChatResponse, TodoRead
from .tools.todos import list_todos
from .langchain_agent import run_agent
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


@app.post("/chat-langchain", response_model=ChatResponse)
async def chat_langchain(
    payload: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """Chat endpoint that uses LangChain for RAG + tool calling."""
    if settings.api_token and payload.api_token != settings.api_token:
        raise HTTPException(status_code=401, detail="Invalid API token")

    reply, used_tools, retrieved_ids = await run_agent(payload.message)
    return ChatResponse(reply=reply, used_tools=used_tools, retrieved_doc_ids=retrieved_ids)


@app.get("/todos", response_model=List[TodoRead])
def get_todos(
    status: Optional[str] = None,
    db: Session = Depends(get_db),
) -> List[TodoRead]:
    """Return the list of todos, optionally filtered by status."""
    return list_todos(db, status=status)


# Serve the simple frontend under /ui to avoid conflicting with API endpoints
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="static")
