# chat-with-llm

A minimal Gradio chat UI powered by OpenAI GPT-5, managed with Poetry.

## Prerequisites

- Python 3.9–3.12
- Poetry (`pipx install poetry` or `pip install --user poetry`)

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

## Notes

- The app uses the environment variable `OPENAI_API_KEY` loaded from `.env` via `python-dotenv`.
- The model is set to `gpt-5`. You can change it in `app/main.py` if desired.
