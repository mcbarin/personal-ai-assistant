"""
DEPRECATED: This module is no longer used.

The application now uses LangChain-based agent in langchain_agent.py
with MCP client integration via mcp_clients/ registry.

This file is kept for reference but should not be imported or used.
"""
from datetime import datetime, timedelta
import json
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from .db import SessionLocal
from .llm.ollama import OllamaProvider
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

    human = _format_human_datetime_range(start, end)
    reply = (
        f"Created calendar event '{title}' for {human}.\n"
        f"Go to calendar event: {link}"
    )
    return reply, ["create_event"], []


async def _classify_intent(message: str) -> str:
    """Use the LLM to classify the user's intent.

    Returns one of: 'TODO', 'EVENT', 'QA'.
    """
    system_prompt = (
        "You are an intent classifier for a personal assistant.\n"
        "Given a single user message, decide if the primary intent is:\n"
        "- TODO: creating or updating a personal todo/reminder/task.\n"
        "- EVENT: scheduling or modifying a calendar event/meeting.\n"
        "- QA: asking a question or chatting (no tool call).\n"
        "Reply with exactly one word: TODO, EVENT, or QA."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    llm = OllamaProvider()
    raw = await llm.generate(messages)
    intent = raw.strip().upper().split()[0]
    if intent not in {"TODO", "EVENT", "QA"}:
        return "QA"
    return intent


def _handle_todo_nl(db: Session, message: str) -> Tuple[str, List[str], List[str]]:
    """Create a todo directly from a natural-language message.

    MVP behaviour: store the full message as the todo text with no due date.
    """
    todo = todo_tools.create_todo(db, text=message.strip(), due_at=None)
    reply = (
        f"I created a todo from that message.\n"
        f"- ID: {todo.id}\n"
        f"- Text: {todo.text}\n"
        f"- Due: no due date\n\n"
        f"You can also use the explicit format "
        f"'todo: Buy milk | 2025-11-15' if you want to include a due date."
    )
    return reply, ["create_todo"], []


async def _extract_event_details(message: str) -> Optional[Tuple[str, datetime, datetime]]:
    """Use the LLM to extract structured event details from natural language.

    The model is asked to return JSON with:
    - title: short event title
    - start: ISO 8601 datetime (e.g. 2025-11-15T09:00:00)
    - end: ISO 8601 datetime (or omitted; then we default to +1 hour)
    """
    now = datetime.utcnow()
    today_str = now.date().isoformat()

    system_prompt = (
        "You extract calendar event details from natural language.\n"
        "Given one user message, output ONLY a JSON object with keys:\n"
        '{ "title": string, "start": string, "end": string }.\n'
        "- 'start' and 'end' must be full ISO 8601 datetimes (e.g. 2025-11-15T09:00:00).\n"
        f"- Today is {today_str}. Interpret relative dates like 'today', 'tomorrow', or "
        "named weekdays relative to this date.\n"
        "- Assume the user means their local timezone; you don't need to add a timezone suffix.\n"
        "- If the user does not specify an end time, set 'end' to exactly 1 hour after 'start'.\n"
        "Do not include any explanation text, only the JSON."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    llm = OllamaProvider()
    raw = await llm.generate(messages)

    try:
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx == -1 or end_idx == -1:
            return None
        data = json.loads(raw[start_idx : end_idx + 1])
    except Exception:
        return None

    try:
        title = str(data["title"]).strip()
        start_str = str(data["start"]).strip()
        start_dt = datetime.fromisoformat(start_str)
        end_str = data.get("end")
        if end_str:
            end_dt = datetime.fromisoformat(str(end_str).strip())
        else:
            end_dt = start_dt + timedelta(hours=1)
        return title, start_dt, end_dt
    except Exception:
        return None


async def _handle_event_nl(message: str) -> Tuple[str, List[str], List[str]]:
    """Handle natural-language event requests using LLM-based extraction."""
    details = await _extract_event_details(message)
    if not details:
        reply = (
            "This looks like a calendar event request, but I couldn't confidently extract the "
            "date and time.\n\n"
            "Please try again with a bit more explicit phrasing like "
            "'Schedule coffee with John tomorrow at 3pm for 1 hour' or "
            "'Schedule coffee with John on 2025-11-15 from 09:00 to 10:00'."
        )
        return reply, [], []

    title, start, end = details
    event = calendar_tools.create_event(title=title, start=start, end=end, description=None)
    link = event.get("htmlLink", "(no link)")

    human = _format_human_datetime_range(start, end)
    reply = (
        f"Created calendar event '{title}' for {human}.\n"
        f"Go to calendar event: {link}"
    )
    return reply, ["create_event"], []


def _format_human_datetime_range(start: datetime, end: datetime) -> str:
    """Return a human-friendly description like 'tomorrow, 11pm–12am'."""
    now = datetime.now()
    start_date = start.date()
    today = now.date()
    tomorrow = today + timedelta(days=1)

    if start_date == today:
        date_label = "today"
    elif start_date == tomorrow:
        date_label = "tomorrow"
    else:
        date_label = start.strftime("%b %-d")  # e.g. 'Nov 15'

    def fmt_time(dt: datetime) -> str:
        # e.g. '11:00pm' or '3pm'
        hour = dt.strftime("%-I").lstrip("0")
        minute = dt.strftime("%M")
        ampm = dt.strftime("%p").lower()
        if minute == "00":
            return f"{hour}{ampm}"
        return f"{hour}:{minute}{ampm}"

    start_str = fmt_time(start)
    end_str = fmt_time(end)
    return f"{date_label}, {start_str}–{end_str}"


async def handle_message(message: str) -> ChatResponse:
    """Main entrypoint for handling user messages.

    MVP behaviour:
    - If the message starts with 'todo:' or 'event:', call the corresponding tool directly.
    - Otherwise, use the LLM to classify the intent:
      - TODO: create a todo from the message text.
      - EVENT: explain how to create an event using the explicit syntax.
      - QA: run the RAG pipeline and answer based on your notes.
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
            intent = await _classify_intent(message_stripped)
            if intent == "TODO":
                reply, used_tools, retrieved_ids = _handle_todo_nl(db, message_stripped)
            elif intent == "EVENT":
                reply, used_tools, retrieved_ids = await _handle_event_nl(message_stripped)
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
