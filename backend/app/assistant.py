from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from .db import SessionLocal
from .models import ConversationLog
from .rag.pipeline import answer_with_context
from .schemas import ChatResponse
from .tools import calendar as calendar_tools
from .tools import todos as todo_tools


def _parse_datetime(value: str) -> datetime:
    """Parse a simple datetime string.

    Supported formats (UTC assumed):
    - YYYY-MM-DD
    - YYYY-MM-DD HH:MM
    - YYYY-MM-DDTHH:MM
    """
    value = value.strip()
    # Date only
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass

    # Try adding seconds if missing
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise ValueError(f"Could not parse datetime from '{value}'. Use ISO format like '2025-11-14 17:30'.")


def _handle_todo_command(db: Session, message: str) -> Tuple[str, List[str], List[str]]:
    """Handle messages starting with 'todo:'.

    Syntax (MVP):
    - 'todo: Buy milk'
    - 'todo: Pay rent | 2025-11-15'
    """
    body = message[len("todo:") :].strip()
    if "|" in body:
        text_part, due_part = [p.strip() for p in body.split("|", 1)]
        due_at: Optional[datetime]
        if due_part:
            due_at = _parse_datetime(due_part)
        else:
            due_at = None
    else:
        text_part = body
        due_at = None

    todo = todo_tools.create_todo(db, text=text_part, due_at=due_at)
    due_str = todo.due_at.isoformat() if todo.due_at else "no due date"
    reply = f"Created todo #{todo.id}: '{todo.text}' (due: {due_str})."
    return reply, ["create_todo"], []


def _handle_event_command(message: str) -> Tuple[str, List[str], List[str]]:
    """Handle messages starting with 'event:'.

    Syntax (MVP):
    - 'event: Coffee with John | 2025-11-15 09:00 | 2025-11-15 10:00'
    """
    body = message[len("event:") :].strip()
    parts = [p.strip() for p in body.split("|")]
    if len(parts) < 3:
        raise ValueError(
            "Invalid event syntax. Use: event: Title | 2025-11-15 09:00 | 2025-11-15 10:00"
        )
    title, start_str, end_str = parts[:3]
    start = _parse_datetime(start_str)
    end = _parse_datetime(end_str)

    event = calendar_tools.create_event(title=title, start=start, end=end, description=None)
    link = event.get("htmlLink", "(no link)")
    reply = f"Created calendar event '{title}' from {start.isoformat()} to {end.isoformat()}.\nLink: {link}"
    return reply, ["create_event"], []


async def handle_message(message: str) -> ChatResponse:
    """Main entrypoint for handling user messages.

    MVP behaviour:
    - If the message starts with 'todo:' or 'event:', call the corresponding tool directly.
    - Otherwise, run the RAG pipeline and answer based on your notes.
    """
    used_tools: List[str] = []
    retrieved_ids: List[str] = []

    # Normalize for command detection but keep original for logging
    message_stripped = message.strip()
    lower = message_stripped.lower()

    db: Session = SessionLocal()
    try:
        if lower.startswith("todo:"):
            reply, used_tools, retrieved_ids = _handle_todo_command(db, message_stripped)
        elif lower.startswith("event:"):
            reply, used_tools, retrieved_ids = _handle_event_command(message_stripped)
        else:
            reply, retrieved_ids = await answer_with_context(message)

        log = ConversationLog(
            user_message=message,
            assistant_reply=reply,
            tools_used=",".join(used_tools) if used_tools else None,
            retrieved_doc_ids=",".join(retrieved_ids) if retrieved_ids else None,
        )
        db.add(log)
        db.commit()
    finally:
        db.close()

    return ChatResponse(
        reply=reply,
        used_tools=used_tools,
        retrieved_doc_ids=retrieved_ids,
    )


