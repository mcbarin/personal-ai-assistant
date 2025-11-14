from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    api_token: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    used_tools: List[str] = []
    retrieved_doc_ids: List[str] = []


class TodoCreate(BaseModel):
    text: str
    due_at: Optional[datetime] = None


class TodoRead(BaseModel):
    id: int
    text: str
    due_at: Optional[datetime] = None
    status: str
    created_at: datetime

    class Config:
        orm_mode = True


