"""Gradio app entry point for the RAG study assistant."""

from __future__ import annotations

import html
import re
import time
from urllib.parse import quote

import gradio as gr

from chain import answer_query
from retriever import get_retriever


retriever = get_retriever()


def _build_sources_block(sources: dict[str, str]) -> str:
    if not sources:
        return ""
    lines = ["", "Sources:"]
    for citation, src in sources.items():
        lines.append(f"- {citation} {src}")
    return "\n".join(lines)


def _strip_sources_section(text: str) -> str:
    return re.sub(r"\n{0,2}Sources:\s*[\s\S]*$", "", text, flags=re.IGNORECASE).strip()


def _extract_key_points(answer: str, max_points: int = 5) -> list[str]:
    clean = _strip_sources_section(answer)
    clean = re.sub(r"[*_`#>-]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    points = []
    for s in sentences:
        if len(s) < 24:
            continue
        points.append(s)
        if len(points) >= max_points:
            break
    return points or [clean[:180] + ("..." if len(clean) > 180 else "")]


def _wrap_svg_line(text: str, width: int = 58) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if len(trial) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _build_summary_svg_html(question: str, answer: str) -> str:
    points = _extract_key_points(answer)
    y = 92
    line_height = 28
    bullet_gap = 12
    text_lines: list[tuple[str, int]] = []

    for point in points:
        wrapped = _wrap_svg_line(point)
        if not wrapped:
            continue
        text_lines.append((f"• {wrapped[0]}", y))
        y += line_height
        for line in wrapped[1:]:
            text_lines.append((f"  {line}", y))
            y += line_height
        y += bullet_gap

    content_height = max(340, y + 24)
    safe_question = html.escape(question[:90] + ("..." if len(question) > 90 else ""))

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="920" height="{content_height}" viewBox="0 0 920 {content_height}">',
        "<defs>",
        '<linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">',
        '<stop offset="0%" stop-color="#0c1222" />',
        '<stop offset="100%" stop-color="#16233f" />',
        "</linearGradient>",
        "</defs>",
        '<rect x="0" y="0" width="920" height="{h}" rx="20" fill="url(#bg)"/>'.format(h=content_height),
        '<text x="36" y="44" fill="#a5b4fc" font-size="18" font-family="Menlo, Monaco, monospace">Essential Summary</text>',
        f'<text x="36" y="70" fill="#e2e8f0" font-size="16" font-family="Menlo, Monaco, monospace">Q: {safe_question}</text>',
    ]
    for text, y_pos in text_lines:
        svg_lines.append(
            f'<text x="36" y="{y_pos}" fill="#f8fafc" font-size="20" '
            f'font-family="Menlo, Monaco, monospace">{html.escape(text)}</text>'
        )
    svg_lines.append("</svg>")
    svg = "".join(svg_lines)
    return f'<img alt="summary-svg" src="data:image/svg+xml;utf8,{quote(svg)}" style="width:100%;height:auto;border-radius:14px;" />'


def stream_chat(user_message: str, history: list[dict] | None):
    history = list(history or [])
    message = (user_message or "").strip()
    if not message:
        yield history, "", "Enter a question to begin.", ""
        return

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "_Thinking..._"})
    yield history, "", "Loading retrieved context...", ""

    prior_turns = history[:-2]
    answer, sources = answer_query(message, retriever, chat_history=prior_turns)
    answer = answer + _build_sources_block(sources)

    output = ""
    for token in re.findall(r"\S+|\s+", answer):
        output += token
        history[-1]["content"] = output
        yield history, "", "Generating response...", ""
        time.sleep(0.01)

    summary_svg = _build_summary_svg_html(message, answer)
    yield history, "", "Ready.", summary_svg


with gr.Blocks(title="RAG Algorithms Study Assistant") as demo:
    gr.Markdown("# RAG-Based Algorithms Study Assistant")
    gr.Markdown(
        "Ask questions about algorithms and data structures. "
        "Supports chat history, live rendering, and Markdown/LaTeX math."
    )

    chat = gr.Chatbot(
        label="Chat",
        height=520,
        latex_delimiters=[
            {"left": "$$", "right": "$$", "display": True},
            {"left": "$", "right": "$", "display": False},
        ],
    )
    status = gr.Markdown("Ready.")
    summary_card = gr.HTML(label="Summary Snapshot")

    with gr.Row():
        query = gr.Textbox(
            label="Message",
            placeholder="Explain the Master Theorem and when each case applies.",
            lines=1,
            scale=8,
        )
        ask_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear", scale=1)

    ask_btn.click(
        stream_chat,
        inputs=[query, chat],
        outputs=[chat, query, status, summary_card],
        show_progress="full",
    )
    query.submit(
        stream_chat,
        inputs=[query, chat],
        outputs=[chat, query, status, summary_card],
        show_progress="full",
    )

    clear_btn.click(
        lambda: ([], "", "Ready.", ""),
        inputs=None,
        outputs=[chat, query, status, summary_card],
    )


if __name__ == "__main__":
    demo.queue().launch()
