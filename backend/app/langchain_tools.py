from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from .db import SessionLocal
from .tools import calendar as calendar_tools
from .tools import todos as todo_tools


def create_todo_tool(text: str, due_iso: Optional[str] = None) -> str:
    """Create a todo item using the existing DB-backed implementation."""
    db: Session = SessionLocal()
    try:
        due_at: Optional[datetime] = None
        if due_iso:
            try:
                due_at = datetime.fromisoformat(due_iso)
            except ValueError:
                pass

        todo = todo_tools.create_todo(db, text=text, due_at=due_at)
        due_str = todo.due_at.isoformat() if todo.due_at else "no due date"
        return f"Created todo #{todo.id}: '{todo.text}' (due: {due_str})."
    finally:
        db.close()


def create_event_tool(
    title: str,
    start_iso: str,
    end_iso: str,
    description: Optional[str] = None,
) -> str:
    """Create a Google Calendar event using the existing implementation."""
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)

    event = calendar_tools.create_event(
        title=title,
        start=start,
        end=end,
        description=description,
    )
    link = event.get("htmlLink", "(no link)")
    return f"Created event '{title}' from {start_iso} to {end_iso}. Link: {link}"


