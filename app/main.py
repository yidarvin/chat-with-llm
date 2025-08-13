import os
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI


def generate_response(messages: List[dict]) -> str:
    """
    Get a non-streaming assistant response from OpenAI Chat Completions.
    """
    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"[Error] {e}"


def respond(message: str, history: List[Tuple[str, str]]) -> str:
    conversation: List[dict] = []
    for user_msg, assistant_msg in history:
        if user_msg:
            conversation.append({"role": "user", "content": user_msg})
        if assistant_msg:
            conversation.append({"role": "assistant", "content": assistant_msg})
    conversation.append({"role": "user", "content": message})
    return generate_response(conversation)


def build_interface() -> gr.Blocks:
    return gr.ChatInterface(
        fn=respond,
        title="Chat with GPT-5",
        description="A minimal Gradio chat UI powered by OpenAI GPT-5.",
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


