## Personal Assistant (FastAPI + RAG + Google Calendar)

This project is a **personal AI assistant** you can chat with. It:

- Uses a local or hosted **LLM** (Ollama by default) to understand your requests.
- Uses **RAG** over your own notes to answer questions grounded in your data.
- Can **create and list todos** in a local database.
- Can **create real events** in your **Google Calendar**.

The stack is designed to showcase modern AI integration for backend-heavy roles:

- **Backend**: FastAPI
- **AI orchestration**: LangChain-style components (LLM provider abstraction, tools, RAG pipeline)
- **Vector DB**: Qdrant or Chroma
- **DB**: Postgres (via Docker) for todos and logs
- **Infra**: Docker + docker-compose

### High-level architecture

- `FastAPI` app exposes:
  - `POST /chat` – text chat with the assistant.
- **LLM provider abstraction**:
  - Default: local **Ollama** (e.g., Llama 3) via HTTP.
  - Drop-in replacements for OpenAI/Anthropic later.
- **RAG pipeline**:
  - Offline ingestion script indexes files from a local `notes/` folder.
  - Embeds documents and stores them in **Qdrant** (Docker container) or Chroma.
  - At query time, retrieves top-k chunks and feeds them into the LLM as context.
- **Tools / agents**:
  - Todo tools: `create_todo`, `list_todos` (backed by Postgres).
  - Calendar tools: `create_google_calendar_event`, `list_google_calendar_events`.
  - These are wired into an assistant pipeline so that LLM can call them when needed.
- **Persistence**:
  - Postgres stores todos and conversation logs.

### Project layout (backend-focused)

```text
backend/
  app/
    __init__.py
    main.py                # FastAPI app & routes
    config.py              # Settings (LLM provider, DB URL, Google creds, etc.)
    db.py                  # SQLAlchemy engine/session
    models.py              # SQLAlchemy models (Todo, ConversationLog, etc.)
    schemas.py             # Pydantic models for API I/O
    llm/
      __init__.py
      base.py              # LLMProvider interface
      ollama.py            # Ollama implementation
    rag/
      __init__.py
      ingest.py            # CLI for indexing notes/ into vector store
      pipeline.py          # RAG query pipeline used by assistant
    tools/
      __init__.py
      todos.py             # Todo tools (create/list)
      calendar.py          # Google Calendar tools
    assistant.py           # Orchestration: RAG + tools + LLM (single entrypoint)
docker-compose.yml
```

### Step-by-step implementation checklist

You can follow this checklist as you build and run the project.

1. **Start core services with Docker**
   - From the project root:

     ```bash
     cd ~/workspace/tinkering
     docker compose up --build
     ```

   - This runs:
     - `api` (FastAPI),
     - `db` (Postgres),
     - `qdrant` (vector DB).

2. **Initialize database tables (one-time per fresh DB volume)**
   - In a separate terminal:

     ```bash
     cd ~/workspace/tinkering
     docker compose exec api python -c \
     "from app.db import engine, Base; import app.models  # noqa: F401; Base.metadata.create_all(bind=engine)"
     ```

   - This creates the `todos` and `conversation_logs` tables.
   - If you ever remove the DB volume (e.g. `docker compose down -v`), you must run this again.

3. **Ingest notes into Qdrant for RAG**
   - Add some markdown/text notes under `backend/notes/`, for example:

     ```bash
     cd ~/workspace/tinkering
     mkdir -p backend/notes
     cat > backend/notes/example.md << 'EOF'
     # Personal goals
     Learn RAG, LangChain, and MCP. Build a personal assistant.
     EOF
     ```

   - Then ingest them inside the container:

     ```bash
     cd ~/workspace/tinkering
     docker compose exec api python -m app.rag.ingest
     ```

   - You should see `Ingested example.md` in the logs.

4. **Test the API**
   - Health check:

     ```bash
     curl http://localhost:8000/health
     ```

   - Test chat with RAG:

     ```bash
     curl -X POST http://localhost:8000/chat \
       -H "Content-Type: application/json" \
       -d '{"message": "What are my personal goals?", "api_token": "dev-token-123"}'
     ```

   - You should get a JSON response whose `reply` reflects the content of your notes.

5. **Initialize Google Calendar OAuth (one-time on your machine)**
   - Install required Google libraries:

     ```bash
     python3 -m pip install --user google-auth-oauthlib google-api-python-client
     ```

   - Place your downloaded OAuth client JSON as:

     ```bash
     mv /path/to/downloaded/client_secret_*.json ~/workspace/tinkering/backend/google_credentials.json
     ```

   - Run the OAuth helper script from the backend folder:

     ```bash
     cd ~/workspace/tinkering/backend
     python3 google_oauth_init.py
     ```

   - This will open a browser for consent and then write `backend/google_token.json`.

6. **Sanity-check Calendar integration (inside Docker)**
   - After you have `google_token.json`, rebuild and start Docker:

     ```bash
     cd ~/workspace/tinkering
     docker compose down
     docker compose up --build
     ```

   - Then run a quick test event from inside the `api` container:

     ```bash
     cd ~/workspace/tinkering
     docker compose exec api python -c "
     from datetime import datetime, timedelta
     from app.tools.calendar import create_event
     now = datetime.utcnow()
     event = create_event(
         title='Test event from assistant',
         start=now + timedelta(minutes=10),
         end=now + timedelta(minutes=40),
         description='Testing Google Calendar integration',
     )
     print('Created event link:', event.get('htmlLink'))
     "
     ```

   - You should see a link and a matching event in your Google Calendar.

7. **Set up Notion MCP (optional, for `/chat-langchain` endpoint)**
   - Create a Notion integration:
     - Go to [Notion Integrations](https://www.notion.so/my-integrations) and create a new internal integration.
     - Copy the integration token (starts with `ntn_`).
     - Create a Todo database in Notion (or use an existing one).
     - Share the database with your integration (click "Share" → "Invite" → select your integration).
     - Copy the database ID from the database URL (the long string after the last `/`).
   - Add to your `.env`:
     ```bash
     NOTION_INTEGRATION_TOKEN=your_token_here
     NOTION_DATABASE_ID=your_database_id_here
     ```
     Note: The code reads `NOTION_INTEGRATION_TOKEN` from `.env` and passes it via `OPENAPI_MCP_HEADERS` environment variable to the Docker MCP server (which is what `mcp/notion` expects for OpenAPI MCP).

   - **Verify your token is valid** by testing it directly:
     ```bash
     curl -X GET 'https://api.notion.com/v1/users/me' \
       -H 'Authorization: Bearer YOUR_TOKEN_HERE' \
       -H 'Notion-Version: 2022-06-28'
     ```
     Replace `YOUR_TOKEN_HERE` with your actual token. If you get a 401, the token is invalid.

   - **Important**: Make sure your Notion database is shared with your integration:
     - Open your database in Notion
     - Click "Share" → "Invite" → Select your integration
     - Without this, you'll get 401 errors even with a valid token
   - The `/chat-langchain` endpoint will automatically use Notion MCP tools when configured.
   - The Notion MCP server runs via Docker (`mcp/notion` image) using `langchain-mcp-adapters`.

8. **Test the LangChain endpoint with Notion**
   - After setting up Notion, test:
     ```bash
     curl -X POST http://localhost:8000/chat-langchain \
       -H "Content-Type: application/json" \
       -d '{"message": "Remind me to learn about MCP tomorrow", "api_token": "dev-token-123"}'
     ```
   - This should create a todo in your Notion database (if `NOTION_TOKEN` is set) or fall back to local DB.

More detailed backend and frontend implementation will live under `backend/` (and optionally `frontend/`) as you build out the project.


