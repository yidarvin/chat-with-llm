import os
import json
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Tuple, Optional

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai


SUPPORTED_MODELS: List[str] = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-3.5-turbo",
    "o3-deep-research",
    "o4-mini-deep-research",
    "o3",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

CHAT_MODELS: List[str] = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-3.5-turbo",
]

RESPONSES_ONLY_MODELS: List[str] = [
    "o3-deep-research",
    "o4-mini-deep-research",
    "o3",
]

ANTHROPIC_MODELS: List[str] = [
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
]

GEMINI_MODELS: List[str] = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


def _messages_to_responses_input(messages: List[dict]) -> List[dict]:
    structured: List[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content_text = m.get("content", "")
        # Map roles to appropriate content block types for the Responses API
        if role == "assistant":
            block_type = "output_text"
            mapped_role = "assistant"
        else:
            # Treat user/system as user input text
            block_type = "input_text"
            mapped_role = "user" if role != "assistant" else "assistant"
        structured.append({
            "role": mapped_role,
            "content": [
                {"type": block_type, "text": content_text}
            ],
        })
    return structured


def generate_response(messages: List[dict], model_name: str) -> str:
    """
    Get a non-streaming assistant response from OpenAI Chat Completions.
    """
    client = OpenAI()
    try:
        if model_name in RESPONSES_ONLY_MODELS:
            resp = client.responses.create(
                model=model_name,
                input=_messages_to_responses_input(messages),
                tools=[{"type": "web_search_preview"}],
            )
            # New SDKs expose convenience: output_text
            text = getattr(resp, "output_text", None)
            if text is None:
                # Fallback: attempt to read from first output
                try:
                    text = resp.output[0].content[0].text  # type: ignore[attr-defined]
                except Exception:
                    text = ""
            return text or ""
        elif model_name in ANTHROPIC_MODELS:
            anthropic = Anthropic()
            # Convert to Anthropic Messages API format
            # Use the most recent user message as the input and include prior turns
            anthro_messages: List[dict] = []
            for m in messages:
                role = m.get("role")
                content = m.get("content", "")
                if role == "system":
                    anthro_messages.append({"role": "system", "content": content})
                elif role == "assistant":
                    anthro_messages.append({"role": "assistant", "content": content})
                else:
                    anthro_messages.append({"role": "user", "content": content})
            resp = anthropic.messages.create(
                model=model_name,
                messages=anthro_messages,
                max_tokens=1024,
            )
            try:
                # Concatenate text blocks
                return "".join(part.text for part in resp.content if getattr(part, "type", "") == "text")
            except Exception:
                return str(resp)
        elif model_name in GEMINI_MODELS:
            # Configure from env var
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return "[Error] GOOGLE_API_KEY is not set in the environment/.env"
            genai.configure(api_key=api_key)
            # Build history excluding the last user turn; send that as the prompt
            history = []
            system_instruction = None
            last_user_text = ""
            if messages:
                # Identify last user message
                for m in reversed(messages):
                    if m.get("role") == "user":
                        last_user_text = m.get("content", "")
                        break
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    system_instruction = content
                elif role == "assistant":
                    history.append({"role": "model", "parts": [content]})
                elif role == "user" and content != last_user_text:
                    history.append({"role": "user", "parts": [content]})

            model = genai.GenerativeModel(model_name if model_name else "gemini-2.5-pro", system_instruction=system_instruction)
            chat = model.start_chat(history=history)
            try:
                resp = chat.send_message(last_user_text)
                return getattr(resp, "text", None) or ""
            except Exception as e:
                return f"[Error] {e}"
        else:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            return resp.choices[0].message.content or ""
    except Exception as e:
        error_text = str(e)
        hint = ""
        if "only supported in v1/responses" in error_text.lower():
            hint = "\nHint: This model only supports the Responses API. Please choose a compatible model or keep this one and we will automatically use the Responses API."
        elif "unsupported" in error_text.lower() or "not found" in error_text.lower():
            hint = "\nHint: This model might not support the selected API. Try another model from the dropdown."
        return f"[Error] {error_text}{hint}"


def generate_response_stream(messages: List[dict], model_name: str) -> Generator[str, None, None]:
    """
    Stream assistant tokens; falls back to non-streaming on unsupported cases.
    """
    client = OpenAI()
    try:
        accumulated_text = ""
        if model_name in RESPONSES_ONLY_MODELS:
            stream = client.responses.stream(
                model=model_name,
                input=_messages_to_responses_input(messages),
                tools=[{"type": "web_search_preview"}],
            )
            for event in stream:
                try:
                    event_type = getattr(event, "type", "")
                    if event_type == "response.output_text.delta":
                        delta_text = getattr(event, "delta", None)
                        if delta_text:
                            accumulated_text += delta_text
                            yield accumulated_text
                    elif event_type == "response.completed":
                        break
                except Exception:
                    continue
        elif model_name in ANTHROPIC_MODELS:
            anthropic = Anthropic()
            # For streaming, use the streaming Messages API
            acc = ""
            with anthropic.messages.stream(
                model=model_name,
                messages=[
                    {"role": (m.get("role") if m.get("role") in ["user", "assistant", "system"] else "user"),
                     "content": m.get("content", "")}
                    for m in messages
                ],
                max_tokens=1024,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta" and getattr(event, "delta", None) and getattr(event.delta, "type", "") == "text_delta":
                        piece = getattr(event.delta, "text", "")
                        if piece:
                            acc += piece
                            yield acc
        elif model_name in GEMINI_MODELS:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                yield "[Error] GOOGLE_API_KEY is not set in the environment/.env"
                return
            genai.configure(api_key=api_key)
            history = []
            system_instruction = None
            last_user_text = ""
            if messages:
                for m in reversed(messages):
                    if m.get("role") == "user":
                        last_user_text = m.get("content", "")
                        break
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    system_instruction = content
                elif role == "assistant":
                    history.append({"role": "model", "parts": [content]})
                elif role == "user" and content != last_user_text:
                    history.append({"role": "user", "parts": [content]})
            model = genai.GenerativeModel(model_name if model_name else "gemini-2.5-pro", system_instruction=system_instruction)
            chat = model.start_chat(history=history)
            acc = ""
            try:
                for chunk in chat.send_message(last_user_text, stream=True):
                    if getattr(chunk, "text", None):
                        acc += chunk.text
                        yield acc
            except Exception as e:
                yield f"[Error] {e}"
                return
        else:
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                try:
                    content_piece = chunk.choices[0].delta.content
                    if content_piece:
                        accumulated_text += content_piece
                        yield accumulated_text
                except Exception:
                    continue
    except Exception:
        # Fallback to a single-shot response if streaming is not supported
        fallback = generate_response(messages, model_name)
        if fallback:
            yield fallback


LOG_DIR: Path = Path.cwd() / "chatlogs"
LOG_PATH: Optional[Path] = None


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _get_or_create_log_file() -> Path:
    global LOG_PATH
    _ensure_log_dir()
    if LOG_PATH is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        LOG_PATH = LOG_DIR / f"{ts}.md"
        LOG_PATH.touch(exist_ok=False)
    return LOG_PATH


def _append_log_line(text: str) -> None:
    log_file = _get_or_create_log_file()
    with log_file.open("a", encoding="utf-8") as f:
        f.write(text)


def _append_user_message(content: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _append_log_line(f"\n\n## [{timestamp}] Human\n\n{content}\n")


def _append_assistant_message(content: str, model_name: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _append_log_line(f"\n\n## [{timestamp}] AI ({model_name})\n\n{content}\n")


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def _write_text(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def _parse_front_matter(text: str) -> Tuple[Optional[str], str]:
    """Return (front_matter_block, body) where front_matter_block excludes trailing '---'."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            fm = text[4:end]
            body = text[end + 5 :]
            return fm, body
    return None, text


def _ensure_front_matter(created_date: str, model_name: str) -> None:
    log_file = _get_or_create_log_file()
    current = _read_text(log_file)
    fm, body = _parse_front_matter(current)
    if fm is None:
        import json as _json
        base_tags = ["LLM Chat", model_name]
        tags_yaml = "\n".join([f"  - {_json.dumps(t)}" for t in base_tags])
        fm_block = (
            f"---\n"
            f"title: \"\"\n"
            f"tags:\n{tags_yaml}\n"
            f"created: \"{created_date}\"\n"
            f"modified: \"{created_date}\"\n"
            f"---\n"
        )
        _write_text(log_file, fm_block + body)


def _update_front_matter(title: Optional[str], tags: Optional[List[str]], created_fallback: str, model_name: str) -> None:
    log_file = _get_or_create_log_file()
    current = _read_text(log_file)
    fm, body = _parse_front_matter(current)
    created = created_fallback
    if fm is not None:
        # crude parse of existing fields
        for line in fm.splitlines():
            if line.strip().startswith("created:"):
                # value inside quotes
                idx = line.find('"')
                if idx != -1:
                    created = line[idx + 1 : line.find('"', idx + 1)] or created_fallback
                else:
                    # value may be after colon without quotes
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        created = parts[1].strip().strip('"') or created_fallback
                break
    # sanitize and JSON-escape for YAML values
    safe_title = title or ""
    safe_title_json = json.dumps(safe_title)
    # Merge base tags with summary-proposed tags, de-duplicated while preserving order
    merged_tags: List[str] = []
    for t in ["LLM Chat", model_name] + (tags or []):
        st = str(t)
        if st not in merged_tags:
            merged_tags.append(st)
    tags_block = "\n".join([f"  - {json.dumps(t)}" for t in merged_tags])
    modified = datetime.now().strftime("%Y-%m-%d")
    new_fm_block = (
        f"---\n"
        f"title: {safe_title_json}\n"
        f"tags:\n{tags_block}\n"
        f"created: \"{created}\"\n"
        f"modified: \"{modified}\"\n"
        f"---\n"
    )
    _write_text(log_file, new_fm_block + body)


def _summarize_title_and_tags(messages: List[dict]) -> Tuple[Optional[str], Optional[List[str]]]:
    """Use OpenAI gpt-3.5-turbo to create a short title and 3-7 relevant tags."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None
    try:
        client = OpenAI()
        # Limit to last 24 turns to control prompt size
        trimmed = messages[-24:]
        prompt = (
            "Summarize the chat into a concise title (<= 8 words) and 3-7 lowercase tags.\n"
            "Respond ONLY in JSON with keys: title (string), tags (string array)."
        )
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(trimmed)},
            ],
            temperature=0,
        )
        content = completion.choices[0].message.content or "{}"
        import json as _json

        data = _json.loads(content)
        title = data.get("title")
        tags = data.get("tags")
        if isinstance(tags, list):
            tags = [str(t) for t in tags]
        else:
            tags = None
        return (str(title) if title else None), tags
    except Exception:
        return None, None


def respond(message: str, history: List[Tuple[str, str]], model_name: str) -> Generator[str, None, None]:
    conversation: List[dict] = []
    for user_msg, assistant_msg in history:
        if user_msg:
            conversation.append({"role": "user", "content": user_msg})
        if assistant_msg:
            conversation.append({"role": "assistant", "content": assistant_msg})
    conversation.append({"role": "user", "content": message})
    selected_model = model_name if model_name in SUPPORTED_MODELS else SUPPORTED_MODELS[0]
    # Logging: ensure YAML front-matter exists and write the user message
    created_date = datetime.now().strftime("%Y-%m-%d")
    _ensure_front_matter(created_date, selected_model)
    _append_user_message(message)

    # Stream while accumulating for final write
    accumulated = ""
    for chunk in generate_response_stream(conversation, selected_model):
        accumulated = chunk
        yield chunk

    if accumulated:
        _append_assistant_message(accumulated, selected_model)
        # Update YAML header with summarized title and tags
        title, tags = _summarize_title_and_tags(conversation + [{"role": "assistant", "content": accumulated}])
        _update_front_matter(title, tags, created_date, selected_model)


def build_interface() -> gr.Blocks:
    model_selector = gr.Dropdown(
        label="Model",
        choices=SUPPORTED_MODELS,
        value="gpt-5",
    )
    return gr.ChatInterface(
        fn=respond,
        title="Chat with LLM",
        description="A minimal Gradio chat UI powered by OpenAI with selectable models.",
        additional_inputs=[model_selector],
    )


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Create a .env file with your key.")
    demo = build_interface()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()


