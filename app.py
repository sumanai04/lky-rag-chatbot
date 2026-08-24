"""Gradio chat UI for the Lee Kuan Yew persona chatbot.

Run:
    python app.py
then open http://localhost:7860
"""

import os

from dotenv import load_dotenv

load_dotenv()

import gradio as gr  # noqa: E402

from rag import LKYChatbot  # noqa: E402

bot = LKYChatbot()

EXAMPLES = [
    "What did you feel when Singapore separated from Malaysia in 1965?",
    "What makes a leader worth following?",
    "Why does Singapore need foreign talent?",
    "What did you tell the US Congress about America's role in Asia?",
    "Is it acceptable for a government to interfere in private lives?",
    "What is your view of India's potential?",
]

DESCRIPTION = (
    "A retrieval augmented generation chatbot that answers as Lee Kuan Yew. "
    "Every answer is grounded in a corpus of his speeches, interviews, and memoirs. "
    "This is a learning exercise; it imitates his voice and does not claim to be him."
)


def respond(message, history):  # noqa: ARG001 (history kept for Gradio signature)
    return bot.answer(message)


demo = gr.ChatInterface(
    fn=respond,
    title="Lee Kuan Yew (RAG Persona Chatbot)",
    description=DESCRIPTION,
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))
