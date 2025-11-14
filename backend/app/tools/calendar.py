from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from ..config import get_settings


def _get_service():
    settings = get_settings()
    creds = Credentials.from_authorized_user_file(settings.google_token_file, scopes=["https://www.googleapis.com/auth/calendar"])
    service = build("calendar", "v3", credentials=creds)
    return service


def create_event(
    title: str,
    start: datetime,
    end: datetime,
    description: str | None = None,
) -> Dict[str, Any]:
    """Create an event in Google Calendar using stored OAuth token."""
    service = _get_service()
    settings = get_settings()

    event_body = {
        "summary": title,
        "description": description or "",
        "start": {"dateTime": start.isoformat(), "timeZone": "UTC"},
        "end": {"dateTime": end.isoformat(), "timeZone": "UTC"},
    }

    event = (
        service.events()
        .insert(calendarId=settings.google_calendar_id, body=event_body)
        .execute()
    )
    return event


def list_events(time_min: datetime, time_max: datetime, max_results: int = 10) -> List[Dict[str, Any]]:
    service = _get_service()
    settings = get_settings()

    events_result = (
        service.events()
        .list(
            calendarId=settings.google_calendar_id,
            timeMin=time_min.isoformat() + "Z",
            timeMax=time_max.isoformat() + "Z",
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    return events_result.get("items", [])


