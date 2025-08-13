# chat-with-llm

A minimal Gradio chat UI powered by different LLM's, managed with Poetry.

## Prerequisites

- Python 3.9–3.12
- Poetry (`p` or `pip install --user poetry`)

## Setup

1) Clone and enter the project directory

```bash
git clone <this-repo-url>
cd chat-with-llm
```

2) Create a `.env` file with your OpenAI API key

```bash
cp .env.example .env # if example is available, or create manually
```

The `.env` file must contain:

```bash
OPENAI_API_KEY=sk-your-api-key-here
```

You can create an API key from your OpenAI account settings.

3) Install dependencies with Poetry

```bash
poetry install
```

4) Run the app

```bash
poetry run chat-with-llm
```

Gradio will print a local URL you can open in your browser.

## Docker

Build and run with Docker Compose (recommended):

```bash
docker-compose up --build
```

Environment variables from `.env` are loaded automatically. The app binds to `0.0.0.0:7860` in the container and is exposed on your host at `http://localhost:7860`.

To customize where logs are written in the container, set `CHATLOGS_DIR` (defaults to `/app/chatlogs`). The compose file mounts your local `./chatlogs` into that path.

Chat logs are persisted to your local `chatlogs/` directory via a bind mount.

### Change where chat logs are stored

You can store chat logs in a different directory on your host (e.g., `/path/to/chatlogs`).

- Docker Compose:

  1. Edit `docker-compose.yml` and change the bind mount under `volumes`:

     ```yaml
     services:
       chat:
         volumes:
           - /path/to/chatlogs:/app/chatlogs
     ```

  2. Make sure the host directory exists and is writable:

     ```bash
     mkdir -p /path/to/chatlogs
     ```

  3. Recreate the stack:

     ```bash
     docker compose down
     docker compose up --build
     ```

- Poetry (non-Docker):

  Set `CHATLOGS_DIR` to an absolute path (via shell or `.env`).

  - One-off:

    ```bash
    CHATLOGS_DIR=/path/to/chatlogs poetry run chat-with-llm
    ```

  - Persist via `.env`:

    ```bash
    echo 'CHATLOGS_DIR=/path/to/chatlogs' >> .env
    poetry run chat-with-llm
    ```

On macOS/Windows with Docker Desktop, ensure file sharing is enabled for the chosen path.

## Notes

- The app uses the environment variable `OPENAI_API_KEY` loaded from `.env` via `python-dotenv`.
- The model is set to `gpt-5`. You can change it in `app/main.py` if desired.

## Anthropic (Claude) support

To enable Anthropic models (e.g., Claude Opus 4.1), add the following to your `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-api-key-here
```

Then select `claude-opus-4-1-20250805` in the model dropdown.

## Google Gemini support

To enable Google Gemini (e.g., `gemini-2.5-pro`), add the following to your `.env`:

```bash
GOOGLE_API_KEY=your-google-genai-api-key
```

Then select `gemini-2.5-pro`, `gemini-2.5-flash`, or `gemini-2.5-flash-lite` in the model dropdown.
