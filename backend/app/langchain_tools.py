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


def _format_human_datetime_range(start: datetime, end: datetime) -> str:
    """Return a human-friendly description like 'tomorrow, 11pm–12am'."""
    from datetime import timedelta
    now = datetime.now()
    start_date = start.date()
    today = now.date()
    tomorrow = today + timedelta(days=1)

    if start_date == today:
        date_label = "today"
    elif start_date == tomorrow:
        date_label = "tomorrow"
    else:
        # Format as 'Nov 15' - use day without leading zero
        day = start.day
        date_label = start.strftime(f"%b {day}")  # e.g. 'Nov 15'

    def fmt_time(dt: datetime) -> str:
        # e.g. '11:00pm' or '3pm'
        hour = dt.strftime("%I").lstrip("0") or "12"  # Handle hour 0 (midnight)
        minute = dt.strftime("%M")
        ampm = dt.strftime("%p").lower()
        if minute == "00":
            return f"{hour}{ampm}"
        return f"{hour}:{minute}{ampm}"

    start_str = fmt_time(start)
    end_str = fmt_time(end)
    return f"{date_label}, {start_str}–{end_str}"


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
    link = event.get("htmlLink")
    if link:
        human_time = _format_human_datetime_range(start, end)
        return f"Created event '{title}' for {human_time}.\nGo to Calendar: {link}"
    return f"Created event '{title}'"
