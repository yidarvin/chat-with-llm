import os
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI


SUPPORTED_MODELS: List[str] = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "o3-deep-research",
    "o4-mini-deep-research",
    "o3",
]


def generate_response(messages: List[dict], model_name: str) -> str:
    """
    Get a non-streaming assistant response from OpenAI Chat Completions.
    """
    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        error_text = str(e)
        hint = ""
        if "unsupported" in error_text.lower() or "not found" in error_text.lower():
            hint = "\nHint: This model might not support the Chat Completions API. Try another model from the dropdown."
        return f"[Error] {error_text}{hint}"


def respond(message: str, history: List[Tuple[str, str]], model_name: str) -> str:
    conversation: List[dict] = []
    for user_msg, assistant_msg in history:
        if user_msg:
            conversation.append({"role": "user", "content": user_msg})
        if assistant_msg:
            conversation.append({"role": "assistant", "content": assistant_msg})
    conversation.append({"role": "user", "content": message})
    selected_model = model_name if model_name in SUPPORTED_MODELS else SUPPORTED_MODELS[0]
    return generate_response(conversation, selected_model)


def build_interface() -> gr.Blocks:
    model_selector = gr.Dropdown(
        label="Model",
        choices=SUPPORTED_MODELS,
        value="gpt-5",
    )
    return gr.ChatInterface(
        fn=respond,
        title="Chat with GPT-5",
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


