"""Gradio app entry point for the RAG study assistant."""

from __future__ import annotations

import time

import gradio as gr

from chain import answer_query
from retriever import get_retriever


retriever = get_retriever()


def stream_answer(user_query: str):
    if not user_query or not user_query.strip():
        yield "Please enter a question."
        return

    answer, sources = answer_query(user_query.strip(), retriever)

    output = ""
    for token in answer.split():
        output += token + " "
        yield output
        time.sleep(0.01)

    if sources:
        lines = ["\n\nSources:"]
        for citation, src in sources.items():
            lines.append(f"- {citation} {src}")
        yield output + "\n" + "\n".join(lines)


with gr.Blocks(title="RAG Algorithms Study Assistant") as demo:
    gr.Markdown("# RAG-Based Algorithms Study Assistant")
    gr.Markdown(
        "Ask questions about algorithms and data structures. "
        "Answers are grounded in retrieved textbook passages."
    )

    with gr.Row():
        query = gr.Textbox(
            label="Question",
            placeholder="Explain the Master Theorem and when each case applies.",
            lines=3,
        )

    answer_box = gr.Markdown(label="Answer")

    ask_btn = gr.Button("Ask")
    ask_btn.click(stream_answer, inputs=query, outputs=answer_box)
    query.submit(stream_answer, inputs=query, outputs=answer_box)


if __name__ == "__main__":
    demo.launch()
