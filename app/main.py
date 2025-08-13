import os
from typing import Generator, List, Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic


SUPPORTED_MODELS: List[str] = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "o3-deep-research",
    "o4-mini-deep-research",
    "o3",
    "claude-opus-4-1-20250805",
]

CHAT_MODELS: List[str] = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
]

RESPONSES_ONLY_MODELS: List[str] = [
    "o3-deep-research",
    "o4-mini-deep-research",
    "o3",
]

ANTHROPIC_MODELS: List[str] = [
    "claude-opus-4-1-20250805",
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


def respond(message: str, history: List[Tuple[str, str]], model_name: str) -> Generator[str, None, None]:
    conversation: List[dict] = []
    for user_msg, assistant_msg in history:
        if user_msg:
            conversation.append({"role": "user", "content": user_msg})
        if assistant_msg:
            conversation.append({"role": "assistant", "content": assistant_msg})
    conversation.append({"role": "user", "content": message})
    selected_model = model_name if model_name in SUPPORTED_MODELS else SUPPORTED_MODELS[0]
    yield from generate_response_stream(conversation, selected_model)


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


