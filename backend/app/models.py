from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from .db import Base


class Todo(Base):
    __tablename__ = "todos"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    due_at = Column(DateTime, nullable=True)
    status = Column(String, nullable=False, default="open")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ConversationLog(Base):
    __tablename__ = "conversation_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(String, nullable=False)
    assistant_reply = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    tools_used = Column(String, nullable=True)  # comma-separated list for simplicity
    retrieved_doc_ids = Column(String, nullable=True)


