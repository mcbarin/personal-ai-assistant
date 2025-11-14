import os
from pathlib import Path

from .pipeline import store_document


def ingest_notes(notes_dir: str = "notes") -> None:
    base = Path(notes_dir)
    if not base.exists():
        print(f"Notes directory {base} does not exist, skipping.")
        return

    for root, _, files in os.walk(base):
        for name in files:
            if not name.lower().endswith((".md", ".txt")):
                continue
            path = Path(root) / name
            text = path.read_text(encoding="utf-8")
            doc_id = str(path.relative_to(base))
            store_document(doc_id, text)
            print(f"Ingested {doc_id}")


if __name__ == "__main__":
    ingest_notes()


