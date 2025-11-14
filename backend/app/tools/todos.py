from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from .. import models


def create_todo(db: Session, text: str, due_at: Optional[datetime]) -> models.Todo:
    todo = models.Todo(text=text, due_at=due_at, status="open")
    db.add(todo)
    db.commit()
    db.refresh(todo)
    return todo


def list_todos(db: Session, status: Optional[str] = None) -> List[models.Todo]:
    query = db.query(models.Todo)
    if status:
        query = query.filter(models.Todo.status == status)
    return query.order_by(models.Todo.created_at.desc()).all()


